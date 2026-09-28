"""How well does the baked texture reproduce keyframes it was not baked from? Held-out PSNR of the textured mesh.

    uv run python experiments/texture_holdout.py RUN:MODEL [RUN:MODEL ...]   (writes paper/figures/texture_holdout.json)

Every 8th keyframe of a model is held out; the atlas is baked from the others (up to 48 spread over the model,
as the export does) under each occlusion variant; the textured mesh is ray-cast from each held-out camera
(its lens's k1 included) at 480 px and compared with the photo where the mesh covers it. Untextured
triangles show their fused vertex colour, as in the export's atlas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [  # (name, occluders, slope tolerance, z-buffer long side)
    ("centroids", "centroids", False, 256),  # the baker until 2026-09-28
    ("centroids+slope", "centroids", True, 256),  # the baker since: fewer untextured triangles, the same PSNR
    ("samples+slope", "samples", True, 256),
    ("samples+slope@512", "samples", True, 512),
]
LONG = 480


def _views(rec, ims, images: Path):  # type: ignore[no-untyped-def]
    from torchvision.io import read_file

    from drone3d.export.texture_gpu import View
    from drone3d.gpu.nvjpeg import decode_jpeg

    out = []
    for im in ims:
        img = decode_jpeg(read_file(str(images / im.name)), device="cuda").permute(1, 2, 0).contiguous()
        cam = rec.cameras[im.camera_id]
        s = img.shape[1] / cam.width
        pose = im.cam_from_world()
        k1 = float(cam.params[3]) if cam.model.name in ("SIMPLE_RADIAL", "RADIAL") else 0.0
        out.append(View(cam.params[0] * s, cam.params[1] * s, cam.params[2] * s, np.asarray(pose.rotation.matrix()),
                        np.asarray(pose.translation), img, k1))  # fmt: skip
    return out


def _render(scene, f_uv: np.ndarray, albedo: np.ndarray, rec, im, images: Path):  # type: ignore[no-untyped-def]
    """Ray-cast the textured mesh from ``im``'s camera at LONG px -> (render, photo, hit mask)."""
    import open3d.core as o3c
    import torch
    from PIL import Image

    from drone3d.fastsfm.dense import _undistort

    cam = rec.cameras[im.camera_id]
    s = LONG / max(cam.width, cam.height)
    w, h = round(cam.width * s), round(cam.height * s)
    f, cx, cy = cam.params[0] * s, cam.params[1] * s, cam.params[2] * s
    k1 = float(cam.params[3]) if cam.model.name in ("SIMPLE_RADIAL", "RADIAL") else 0.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xn, yn = _undistort(torch.as_tensor((xs + 0.5 - cx) / f), torch.as_tensor((ys + 0.5 - cy) / f), k1)
    d_cam = np.stack([xn.numpy(), yn.numpy(), np.ones_like(xs)], -1).reshape(-1, 3)
    pose = im.cam_from_world()
    r, t = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
    origin = -r.T @ t
    d = d_cam @ r
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.concatenate([np.broadcast_to(origin, d.shape), d], 1).astype(np.float32)
    ans = scene.cast_rays(o3c.Tensor(rays))
    hit = np.isfinite(ans["t_hit"].numpy())
    pid = ans["primitive_ids"].numpy()[hit]
    bu = ans["primitive_uvs"].numpy()[hit]
    w1, w2 = bu[:, 0], bu[:, 1]
    uv = f_uv[pid, 0] * (1 - w1 - w2)[:, None] + f_uv[pid, 1] * w1[:, None] + f_uv[pid, 2] * w2[:, None]
    ah, aw = albedo.shape[:2]
    px = np.clip((uv[:, 0] * aw).astype(int), 0, aw - 1)
    py = np.clip(((1 - uv[:, 1]) * ah).astype(int), 0, ah - 1)
    col = albedo[py, px].astype(np.float64)  # untextured triangles carry their vertex colour in the atlas
    out = np.zeros((h * w, 3))
    out[hit] = col
    photo = np.asarray(Image.open(images / im.name).convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float64)
    return out.reshape(h, w, 3), photo, hit.reshape(h, w)


def _clean(run: Path, k: str, v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """drone3d.export.terrain.clean_model in the export's levelled frame, returned to the SfM frame."""
    from drone3d.export.terrain import clean_model

    fr_ = json.loads((run / "export" / f"model_{k}" / "frame.json").read_text())
    s_, r_, t_ = float(fr_["scale"]), np.asarray(fr_["rotation"]), np.asarray(fr_["translation"])
    dense = json.loads((run / "dense" / "result.json").read_text())
    voxel = next(m["voxel"] for m in dense["models"] if Path(m["model"]).name == k) * s_
    scene = next(m for m in json.loads((run / "export" / "scene.json").read_text())["models"] if m["dir"] == f"model_{k}")
    ve = s_ * v @ r_.T + t_
    window = 0.6 * scene["focus"]["radius"] if scene.get("focus") else 0.1 * float(np.linalg.norm(np.ptp(ve[:, :2], 0)))
    cv, cf, _ = clean_model(ve, f, voxel=voxel, window=window)
    return ((cv - t_) / s_) @ r_, cf


def main() -> None:
    import open3d as o3d
    import pycolmap

    from drone3d.export.stage import _decimate
    from drone3d.export.texture_gpu import bake_soup_texture

    clean = "--clean" in sys.argv  # raw fused mesh against drone3d.export.terrain's clean model, the baker's default
    rows = []
    for spec in [a for a in sys.argv[1:] if not a.startswith("--")]:
        run_name, k = spec.split(":")
        run = ROOT / "outputs" / run_name
        images = run / "dataset" / "images"
        mesh = o3d.io.read_triangle_mesh(str(run / "dense" / f"model_{k}" / "mesh.ply"))
        v, f, vc = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        meshes = [("raw", v, f, vc)]
        if clean:
            cv, cf = _clean(run, k, v, f)
            from scipy.spatial import cKDTree

            _, near = cKDTree(v).query(cv)  # colours for untextured triangles: the nearest raw vertex's
            meshes.append(("clean", cv, cf, vc[near]))
        rec = pycolmap.Reconstruction(str(run / "dataset" / "sparse" / k))
        posed = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if im.has_pose]
        held = posed[4::8]
        rest = [im for im in posed if im not in held]
        cand = [rest[int(round(i))] for i in np.linspace(0, len(rest) - 1, min(48, len(rest)))]
        views = _views(rec, cand, images)
        for label, v, f, vc in meshes:
            if len(f) > 1.25 * 600_000:
                m_ = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
                m_.vertex_colors = o3d.utility.Vector3dVector(vc)
                v, f, vc = _decimate(m_, v, f, vc, 600_000)
            tm = o3d.t.geometry.TriangleMesh()
            tm.vertex.positions = o3d.core.Tensor(v.astype(np.float32))
            tm.triangle.indices = o3d.core.Tensor(f.astype(np.int32))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tm)
            for name, occ, slope, zbuf in (VARIANTS[1:2] if clean else VARIANTS):
                f_uv, albedo, info = bake_soup_texture(v, f, views, size=4096, fallback_rgb=vc, gain=True, zbuf=zbuf,
                                                       occluders=occ, slope=slope)  # fmt: skip
                psnr, cover = [], []
                for im in held:
                    img, photo, hit = _render(scene, f_uv, albedo, rec, im, images)
                    if not hit.any():  # the mesh is not in this view at all (Hanoi): nothing to compare
                        continue
                    mse = float(((img[hit] - photo[hit]) ** 2).mean())
                    psnr.append(10 * np.log10(255.0**2 / max(mse, 1e-9)))
                    cover.append(float(hit.mean()))
                row = {"model": spec, "variant": name, "mesh": label, "held_out": len(held), "candidates": len(views),
                       "psnr_median": round(float(np.median(psnr)), 2) if psnr else None,
                       "psnr_mean": round(float(np.mean(psnr)), 2) if psnr else None, "views_compared": len(psnr),
                       "unseen_share": round(info["unseen_triangles"] / len(f), 4), "coverage": round(float(np.median(cover)), 3)}  # fmt: skip
                rows.append(row)
                print(row, flush=True)
    out = ROOT / "paper" / "figures" / ("clean_holdout.json" if clean else "texture_holdout.json")
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
