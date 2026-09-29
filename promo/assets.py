"""Real-run footage for the promo film's screen slots, from a fast-profile run.

* ``depth_tiles.mp4`` -- 2 x 2 tiles per keyframe: the keyframe, the depth
  triangulated from flow, the depth completed by the calibrated prior, and the
  textured mesh rendered from the same pose;
* ``model_inset.mp4`` and ``flythrough.mp4`` -- the textured mesh circled around
  its subject (or, without one, flown along the drone's own path), ray-cast with
  ``drone3d.export.render_mesh``;
* ``merge.mp4`` -- the video's passes, one colour each, meeting in one model.

    uv run python promo/assets.py outputs/<run> [model index] [--merge-from outputs/<same video, unmerged>]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)

BUILD = ROOT / "promo" / "build"


def _depth_vis(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import cv2

    v = np.log(np.maximum(d, 1e-6))
    img = cv2.applyColorMap((255 * np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d <= 0] = (51, 29, 11)  # empty: the film's navy
    return img


def depth_tiles(run: Path, model: Path, export_dir: Path, out: Path, *, tile=(450, 253), n=6, per_s=1.3) -> Path:
    import cv2
    import torch
    import trimesh

    from drone3d.export.render_mesh import MeshRenderer
    from drone3d.fastsfm.dense_stage import compute_depths
    from drone3d.fastsfm.mono import MonoDepth
    from drone3d.keyframes.flow import RaftFlow

    r = compute_depths(model, run / "dataset" / "images", RaftFlow("raft_large", batch=16), MonoDepth(),
                       long_side=480, gaps=(2, 4, 8, 12), keyframe_stride=1, min_angle_deg=0.5, rel_tol=0.05)  # fmt: skip
    ims, frames = r["ims"], r["frames"].cpu().numpy()
    torch.cuda.empty_cache()
    tm = trimesh.load(export_dir / "mesh_textured.obj", force="mesh", process=False)
    faces = np.asarray(tm.faces)
    rend = MeshRenderer(np.asarray(tm.vertices), faces, np.asarray(tm.visual.uv)[faces], np.asarray(tm.visual.material.image.convert("RGB")))
    fr = json.loads((export_dir / "frame.json").read_text())
    s_, rot, t_ = fr["scale"], np.asarray(fr["rotation"]), np.asarray(fr["translation"])
    import pycolmap

    rec = pycolmap.Reconstruction(str(model))
    valid = np.concatenate([d[d > 0] for d in r["depths"]])
    lo, hi = np.log(np.percentile(valid, [2, 98]))
    tiles = BUILD / "depth_tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    for old in tiles.glob("*.png"):
        old.unlink()
    for k, i in enumerate(np.linspace(0, len(ims) - 1, n).round().astype(int)):
        im = ims[i]
        cam = rec.cameras[im.camera_id]
        sc = tile[0] / cam.width
        render = rend.render(cam.params[0] * sc, cam.params[1] * sc, cam.params[2] * sc,
                             im.cam_from_world().rotation.matrix() @ rot.T, s_ * rot @ im.projection_center() + t_,
                             (tile[0], round(cam.height * sc)))  # fmt: skip
        fit = lambda a: cv2.resize(a, tile, interpolation=cv2.INTER_AREA)  # noqa: E731
        key = fit(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        tri = fit(_depth_vis(r["tri_depths"][i], lo, hi))
        full = fit(_depth_vis(r["depths"][i], lo, hi))
        mesh = fit(cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tiles / f"t{k:03d}.png"), np.vstack([np.hstack([key, tri]), np.hstack([mesh, full])]))
    from drone3d.io.nvdec import ffmpeg_bin

    subprocess.run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{1 / per_s}", "-i",
                    str(tiles / "t%03d.png"), "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "16", str(out)],
                   check=True)  # fmt: skip
    return out


def _look(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """cam_from_world rotation (COLMAP axes: x right, y down, z forward) looking from ``eye`` at ``target``, z up."""
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def _frames_to_mp4(frames_dir: Path, out: Path, fps: int = 30) -> Path:
    from drone3d.export.render_mesh import _encode

    _encode(frames_dir, fps, out)
    return out


def orbit(export_dir: Path, scene_model: dict, out: Path, *, seconds: float = 10.0, size=(1920, 1080), turns: float = 0.6,
          elev_deg: float = 32.0, keep: float = 1.3, dist: float = 1.9, lift: float = 0.15) -> Path:  # fmt: skip
    """The textured model circled around its subject (scene.json's focus), everything beyond ``keep`` x the
    focus radius left out as the explorer's focus does -- the merge shows as a model complete on every side."""
    import shutil

    import cv2
    import trimesh

    from drone3d.export.render_mesh import MeshRenderer

    tm = trimesh.load(export_dir / "mesh_textured.obj", force="mesh", process=False)
    v, faces = np.asarray(tm.vertices), np.asarray(tm.faces)
    c, r = np.asarray(scene_model["focus"]["center"]), float(scene_model["focus"]["radius"])
    cen = v[faces].mean(1)
    near = (np.abs(cen[:, 0] - c[0]) < keep * r) & (np.abs(cen[:, 1] - c[1]) < keep * r)
    faces = faces[near]
    rend = MeshRenderer(v, faces, np.asarray(tm.visual.uv)[faces], np.asarray(tm.visual.material.image.convert("RGB")))
    ground = float(np.percentile(v[np.unique(faces)][:, 2], 20))
    target = np.array([c[0], c[1], ground + lift * r])  # lift: a tall building is framed at its middle
    w, h = size
    f = 0.9 * w
    tmp = out.parent / (out.stem + "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = int(seconds * 30)
    a0 = np.arctan2(*(np.asarray(scene_model["view"]["eye"])[1::-1] - c[1::-1]))  # start where the drone's view starts
    for i in range(n):
        a = a0 + 2 * np.pi * turns * i / max(n - 1, 1)
        d = dist * r
        eye = target + d * np.array([np.cos(a) * np.cos(np.radians(elev_deg)), np.sin(a) * np.cos(np.radians(elev_deg)),
                                     np.sin(np.radians(elev_deg))])  # fmt: skip
        img = rend.render(f, w / 2, h / 2, _look(eye, target), eye, size)
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    _frames_to_mp4(tmp, out)
    shutil.rmtree(tmp)
    return out


STAGGER = 0.7  # seconds between two passes joining the merge animation (the film's shot times its counter to it)
def orbit_completed(run: Path, out: Path, *, seconds: float = 4.0, size=(1920, 1080), turns: float = 0.5,
                    elev_deg: float = 24.0, keep: float = 1.6, dist: float = 2.4, lift: float = 0.45) -> Path:  # fmt: skip
    """``orbit`` of the measured model with its placed generated object (drone3d.generate): the nearer surface
    wins, so what was measured shows, and the sides the flight never saw come from the generated object."""
    import shutil

    import cv2
    import open3d as o3d
    import open3d.core as o3c
    import trimesh

    sm = json.loads((run / "export" / "scene.json").read_text())["models"][0]
    c, r = np.asarray(sm["focus"]["center"]), float(sm["focus"]["radius"])

    def load(path: Path, yup: bool):  # type: ignore[no-untyped-def]
        m = trimesh.load(path, force="mesh", process=False)
        v = np.asarray(m.vertices, np.float64)
        if yup:
            v = v @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T
        f = np.asarray(m.faces)
        mat = m.visual.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        return v, f, np.asarray(m.visual.uv)[f], np.asarray(img.convert("RGB"))

    meas = load(run / "export" / "model_0" / "mesh_textured.obj", False)
    gen = load(run / "export" / "generated" / "object_aligned.glb", True)
    cen = meas[0][meas[1]].mean(1)
    k = (np.abs(cen[:, 0] - c[0]) < keep * r) & (np.abs(cen[:, 1] - c[1]) < keep * r)
    meas = (meas[0], meas[1][k], meas[2][k], meas[3])
    scenes = []
    for v, f, _, _ in (meas, gen):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3c.Tensor(v.astype(np.float32)), o3c.Tensor(f.astype(np.uint32)))
        scenes.append(sc)
    ground = float(np.percentile(meas[0][np.unique(meas[1])][:, 2], 20))
    target = np.array([c[0], c[1], ground + lift * r])
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    tmp = out.parent / (out.stem + "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = int(seconds * 30)
    a0 = np.arctan2(*(np.asarray(sm["view"]["eye"])[1::-1] - c[1::-1]))
    for i in range(n):
        a = a0 + 2 * np.pi * turns * i / max(n - 1, 1)
        eye = target + dist * r * np.array([np.cos(a) * np.cos(np.radians(elev_deg)), np.sin(a) * np.cos(np.radians(elev_deg)),
                                             np.sin(np.radians(elev_deg))])  # fmt: skip
        d = np.stack([(xs + 0.5 - w / 2) / (0.9 * w), (ys + 0.5 - h / 2) / (0.9 * w), np.ones_like(xs)], -1).reshape(-1, 3) @ _look(eye, target)
        rays = o3c.Tensor(np.ascontiguousarray(np.concatenate([np.broadcast_to(eye.astype(np.float32), d.shape), d.astype(np.float32)], 1)))
        ans = [sc.cast_rays(rays) for sc in scenes]
        ts = np.stack([x["t_hit"].numpy() for x in ans])
        best = np.argmin(np.where(np.isfinite(ts), ts, np.inf), 0)
        img = np.tile(np.array([11, 29, 51], np.uint8), (h * w, 1))
        for j, (x, (_, _, uv, alb)) in enumerate(zip(ans, (meas, gen), strict=True)):
            hit = np.isfinite(ts[j]) & (best == j)
            p = x["primitive_ids"].numpy()[hit]
            b = x["primitive_uvs"].numpy()[hit]
            tuv = (1 - b[:, :1] - b[:, 1:]) * uv[p, 0] + b[:, :1] * uv[p, 1] + b[:, 1:] * uv[p, 2]
            ah, aw = alb.shape[:2]
            img[hit] = alb[np.clip(((1 - tuv[:, 1]) * ah).astype(int), 0, ah - 1), np.clip((tuv[:, 0] * aw).astype(int), 0, aw - 1)]
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), cv2.cvtColor(img.reshape(h, w, 3), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    _frames_to_mp4(tmp, out)
    shutil.rmtree(tmp)
    return out


PASS_COLOURS = [(255, 153, 51), (90, 209, 255), (63, 185, 80), (242, 200, 90), (200, 120, 255), (255, 110, 110),
                (120, 230, 210), (240, 240, 240)]  # fmt: skip


def merge_assembly(single_run: Path, out: Path, *, seconds: float = 6.5, size=(900, 506), keep: float = 1.4) -> Path:
    """The passes of a video meeting: each pass's own fused mesh, one colour per pass, flying from apart into
    the place RoMa v2 + Sim(3) put it (``single_run``: a run mapped before merging, its per-pass models)."""
    import shutil

    import cv2
    import open3d as o3d
    import pycolmap

    from drone3d.export.stage import _level, _subject
    from drone3d.fastsfm.merge import find_links, groups_from_links

    sparse = single_run / "dataset" / "sparse"
    recs = {p.name: pycolmap.Reconstruction(str(p)) for p in sorted(sparse.iterdir(), key=lambda p: int(p.name))}
    recs = {k: r for k, r in recs.items() if r.num_reg_images() >= 3}
    names = sorted(recs, key=lambda k: -recs[k].num_reg_images())
    edges, _ = find_links(recs, single_run / "dataset" / "images")
    group = max(groups_from_links(names, edges), key=lambda g: sum(recs[m].num_reg_images() for m in g["members"]))
    parts, cams, axes, rots = [], [], [], []
    for k, (sim_s, sim_r, sim_t) in group["members"].items():
        mesh = o3d.io.read_triangle_mesh(str(single_run / "dense" / f"model_{k}" / "mesh.ply"))
        parts.append((k, sim_s * np.asarray(mesh.vertices) @ sim_r.T + sim_t, np.asarray(mesh.triangles)))
        for im in recs[k].images.values():
            if im.has_pose:
                cams.append(sim_s * sim_r @ im.projection_center() + sim_t)
                rw = np.asarray(im.cam_from_world().rotation.matrix()) @ sim_r.T
                rots.append(rw)
                axes.append(rw[2])
    cams, axes, rots = np.array(cams), np.array(axes), np.array(rots)
    allv = np.concatenate([p[1] for p in parts])
    lev = _level(allv[:: max(1, len(allv) // 200000)], cams, rots)
    sub = _subject(cams @ lev.T, axes @ lev.T)
    parts = [(k, v @ lev.T, f) for k, v, f in parts]
    if sub is not None:
        c, r = sub
    else:
        allv = np.concatenate([p[1] for p in parts])
        c, r = np.median(allv, 0), float(np.linalg.norm(np.percentile(allv, 90, 0) - np.percentile(allv, 10, 0))) / 3
    kept = []
    for i, (_k, v, f) in enumerate(parts):
        cen = v[f].mean(1)
        sel = (np.abs(cen[:, 0] - c[0]) < keep * r) & (np.abs(cen[:, 1] - c[1]) < keep * r)
        if sel.sum() > 500:
            kept.append((i, v, f[sel]))
    import open3d.core as o3c

    ground = float(np.percentile(np.concatenate([v[np.unique(f)] for _, v, f in kept])[:, 2], 20))
    target = np.array([c[0], c[1], ground + 0.1 * r])
    w, h = size
    tmp = out.parent / (out.stem + "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = int(seconds * 30)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    f_px = 0.95 * w
    light = np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9])
    bg = np.array([11, 29, 51], np.float32)
    cols = np.array(PASS_COLOURS, np.float32)
    for fr in range(n):
        t = fr / 30.0
        scene = o3d.t.geometry.RaycastingScene()
        owner = []
        for j, (i, v, f) in enumerate(kept):  # the model builds up pass by pass: each drops into its place in turn
            arrive = (t - 0.3 - STAGGER * j) / 0.9
            if arrive <= 0:
                continue
            e = 1 - (1 - min(arrive, 1.0)) ** 3
            off = (1 - e) * r * np.array([0.0, 0.0, 0.9])
            scene.add_triangles(o3c.Tensor((v + off).astype(np.float32)), o3c.Tensor(f.astype(np.uint32)))
            owner.append(i)
        if not owner:
            cv2.imwrite(str(tmp / f"{fr:05d}.jpg"), np.full((h, w, 3), bg[::-1], np.uint8))
            continue
        a = 0.9 + 0.5 * t / seconds
        eye = target + 3.1 * r * np.array([np.cos(a) * 0.8, np.sin(a) * 0.8, 0.6])
        rot = _look(eye, target)
        d = np.stack([(xs + 0.5 - w / 2) / f_px, (ys + 0.5 - h / 2) / f_px, np.ones_like(xs)], -1).reshape(-1, 3) @ rot
        rays = np.concatenate([np.broadcast_to(eye.astype(np.float32), d.shape), d.astype(np.float32)], 1)
        ans = scene.cast_rays(o3c.Tensor(np.ascontiguousarray(rays)))
        hit = np.isfinite(ans["t_hit"].numpy())
        gid = ans["geometry_ids"].numpy()[hit]
        nrm = ans["primitive_normals"].numpy()[hit]
        shade = 0.35 + 0.65 * np.abs(nrm @ light)
        img = np.tile(bg, (h * w, 1))
        img[hit] = cols[np.array(owner)[gid] % len(cols)] * shade[:, None]
        cv2.imwrite(str(tmp / f"{fr:05d}.jpg"), cv2.cvtColor(img.reshape(h, w, 3).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])  # fmt: skip
    _frames_to_mp4(tmp, out)
    shutil.rmtree(tmp)
    (out.with_suffix(".json")).write_text(json.dumps({"members": sorted(group["members"]), "shown": len(kept),
                                                      "passes_in_video": len(names), "stagger_s": STAGGER, "first_s": 0.3}))  # fmt: skip
    return out


def main() -> None:
    import argparse

    from drone3d.export.render_mesh import render_mesh_flythrough

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("model", type=int, nargs="?", default=0)
    ap.add_argument("--merge-from", type=Path, default=None, help="the same video mapped before merging (per-pass models)")
    ap.add_argument("--only", default="depth,orbit,merge", help="which assets to (re)make")
    args = ap.parse_args()
    run, k, only = args.run, args.model, set(args.only.split(","))
    BUILD.mkdir(parents=True, exist_ok=True)
    sfm = json.loads((run / "sfm" / "result.json").read_text())
    model = Path(sfm["models"][k]["path"])
    export_dir = run / "export" / f"model_{model.name}"
    scene = json.loads((run / "export" / "scene.json").read_text())["models"][k]
    if "depth" in only:
        print("depth tiles ->", depth_tiles(run, model, export_dir, BUILD / "depth_tiles.mp4"), flush=True)
    if "orbit" in only and scene.get("focus"):
        print("inset ->", orbit(export_dir, scene, BUILD / "model_inset.mp4", seconds=6, size=(760, 428), turns=0.35), flush=True)
        print("orbit ->", orbit(export_dir, scene, BUILD / "flythrough.mp4", seconds=10, size=(1920, 1080)), flush=True)
    elif "orbit" in only:
        print("inset ->", render_mesh_flythrough(export_dir, model, BUILD / "model_inset.mp4", seconds=6, size=(760, 428)), flush=True)
        print("fly-through ->", render_mesh_flythrough(export_dir, model, BUILD / "flythrough.mp4", seconds=10, size=(1920, 1080)), flush=True)
    if "merge" in only and args.merge_from:
        print("merge ->", merge_assembly(args.merge_from, BUILD / "merge.mp4"), flush=True)


if __name__ == "__main__":
    main()
