"""A run's model seen from all around its subject: a contact sheet of N headings.

    uv run python tools/orbit_views.py outputs/<run> [--model 0] [--views 8] [--mesh PATH] [--splats]
                                       [--elevation 20] [--distance 1.0] [--out sheet.jpg]

Rows: the textured mesh (``--mesh``: a textured OBJ/GLB in the export frame, default the export's
``mesh_textured.glb``), the same geometry shaded (so holes and loose triangles show), and with ``--splats``
the trained Gaussian splats. The cameras circle the subject the flight aims at (``export.stage._subject``)
at ``--distance`` times the flight's radius, ``--elevation`` degrees above its centre -- including the
headings the flight never flew, which is the point: a complete model looks whole from every one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)


def look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """cam_from_world rotation (OpenCV: x right, y down, z forward) for z-up world."""
    f = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def main() -> None:
    import open3d as o3d
    import open3d.core as o3c
    import pycolmap
    from PIL import Image, ImageDraw

    from drone3d.export.render_mesh import load_parts, render_parts
    from drone3d.export.stage import _subject

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", type=int, default=0)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--mesh", type=Path, default=None)
    ap.add_argument("--splats", action="store_true")
    ap.add_argument("--splat-dir", type=Path, default=None, help="a trained splat run (splats/model_N_360) instead of the splat stage's")
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sfm = json.loads((a.run / "sfm" / "result.json").read_text())
    model_dir = Path(sfm["models"][a.model]["path"])
    name = model_dir.name
    frame = json.loads((a.run / "export" / f"model_{name}" / "frame.json").read_text())
    fs_, fr, ft = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])
    rec = pycolmap.Reconstruction(str(model_dir))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    cams = np.array([im.projection_center() for im in ims])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in ims])
    sub = _subject(cams, axes)
    ce = fs_ * cams @ fr.T + ft  # flight in the export frame
    if sub is None:
        target, radius = ce.mean(0), float(np.median(np.linalg.norm(ce - ce.mean(0), axis=1)))
    else:
        target, radius = fs_ * fr @ sub[0] + ft, fs_ * sub[1]
    mesh_path = a.mesh or a.run / "export" / f"model_{name}" / "mesh_textured.glb"
    parts = load_parts(mesh_path)
    v = np.concatenate([p[0] for p in parts])
    f = np.concatenate([p[1] + sum(len(q[0]) for q in parts[:i]) for i, p in enumerate(parts)])
    # aim at the middle of what stands at the subject: its ground-to-top span within half the radius
    near = np.linalg.norm(v[:, :2] - target[:2], axis=1) < 0.5 * radius
    if near.sum() > 100:
        z0, z1 = np.percentile(v[near, 2], [2, 99.5])
        target = np.array([target[0], target[1], 0.5 * (z0 + z1)])
    w, h = a.size, round(a.size * 9 / 16)
    cam = rec.cameras[ims[0].camera_id]
    fpx = cam.params[0] * w / cam.width
    views = []
    for k in range(a.views):
        az = 2 * np.pi * k / a.views + np.arctan2(*(ce[0, :2] - target[:2])[::-1])  # heading 0 = the first frame's
        d = a.distance * radius
        c = target + d * np.array([np.cos(az) * np.cos(np.radians(a.elevation)), np.sin(az) * np.cos(np.radians(a.elevation)),
                                   np.sin(np.radians(a.elevation))])  # fmt: skip
        views.append((round(np.degrees(az - np.arctan2(*(ce[0, :2] - target[:2])[::-1]))) % 360, c, look_at(c, target)))
    flown = np.degrees(np.unwrap(np.arctan2(ce[:, 1] - target[1], ce[:, 0] - target[0])))
    flown -= flown[0]
    rows = []
    if any(p[3] is not None for p in parts):
        rows.append([render_parts(parts, fpx, R, c, (w, h))[0] for _, c, R in views])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(v.astype(np.float32)), o3c.Tensor(f.astype(np.uint32)))
    shaded = []
    for _, c, R in views:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (np.stack([(xs + 0.5 - w / 2) / fpx, (ys + 0.5 - h / 2) / fpx, np.ones_like(xs)], -1).reshape(-1, 3) @ R).astype(np.float32)
        hit = scene.cast_rays(o3c.Tensor(np.concatenate([np.broadcast_to(c, d.shape).astype(np.float32), d], 1)))
        n = hit["primitive_normals"].numpy()
        ok = np.isfinite(hit["t_hit"].numpy())
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        lam = np.abs((n * dn).sum(1)) * 0.75 + 0.25 * np.clip(n[:, 2], 0, 1)
        img = np.tile(np.array([11, 29, 51], np.uint8), (h * w, 1))
        img[ok] = (np.clip(lam[ok], 0, 1)[:, None] * np.array([235, 225, 205])).astype(np.uint8)
        shaded.append(img.reshape(h, w, 3))
    rows.append(shaded)
    if a.splats or a.splat_dir:
        import os

        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")  # gsplat's JIT build (as drone3d.splat.render)
        import gsplat
        import torch

        from drone3d.splat.render import load_splat_ply

        if a.splat_dir:
            ply, run_dir = sorted(a.splat_dir.rglob("splat*.ply"), key=lambda p: p.stat().st_mtime)[-1], a.splat_dir
        else:
            splat = json.loads((a.run / "splat" / "result.json").read_text())["models"][a.model]
            ply, run_dir = Path(splat["splat_ply"]), Path(splat["run_dir"])
        spl = load_splat_ply(ply)
        tf = json.loads((run_dir / "scene_transform.json").read_text())["train_from_world"]
        s_tw, r_tw, t_tw = float(tf["scale"]), np.array(tf["rotation"]["matrix_3x3"]), np.array(tf["translation"])
        vms = []
        for _, c, R in views:  # export frame -> SfM frame -> training frame
            c_s = fr.T @ ((c - ft) / fs_)
            r_s = R @ fr
            rc = r_s @ r_tw.T
            vm = np.eye(4)
            vm[:3, :3] = rc
            vm[:3, 3] = s_tw * (-r_s @ c_s) - rc @ t_tw
            vms.append(vm)
        K = torch.tensor([[fpx, 0, w / 2], [0, fpx, h / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            img, _, _ = gsplat.rasterization(spl["means"], spl["quats"], spl["scales"], spl["opacities"], spl["sh"],
                                             torch.tensor(np.array(vms), dtype=torch.float32, device="cuda"),
                                             K[None].expand(len(vms), 3, 3), w, h, sh_degree=spl["sh_degree"],
                                             render_mode="RGB", near_plane=0.01)  # fmt: skip
        rows.append(list((img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()))
    sheet = Image.new("RGB", (w * len(views), h * len(rows)))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            tile = Image.fromarray(im)
            az = views[j][0]
            seen = bool(((flown % 360 - az + 180) % 360 - 180).__abs__().min() < 360 / a.views / 2) if len(flown) else False
            ImageDraw.Draw(tile).text((6, 4), f"{az:3d} deg {'(flown)' if seen else '(never flown)'}", fill=(255, 230, 0))
            sheet.paste(tile, (j * w, i * h))
    out = a.out or a.run / "export" / f"orbit_views_m{name}.jpg"
    sheet.save(out, quality=88)
    print(out, f"flight spans {abs(flown[-1] - flown[0]):.0f} deg of heading" if len(flown) else "")


if __name__ == "__main__":
    main()
