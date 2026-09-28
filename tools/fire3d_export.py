"""A run's model as a Fire3D scene (its ScanNet++ route: posed RGB-D frames) -> ``<out>/scenes/<name>/``.

    uv run python tools/fire3d_export.py outputs/<run> [model] [--out outputs/<run>/fire3d/input] [--size 960]

Fire3D (github.com/xiahongchi/Fire3D, MIT) turns a posed RGB-D video into complete, textured object meshes: it
finds the objects, then generates each one whole (TRELLIS.2 flow models) from every frame that sees it. The
pipeline already has everything it reads:

- ``rgb/NNNNN.png``: the model's keyframes (at most 300, Fire3D's cap), at ``--size`` px wide;
- ``depth_pi3x_conf_chunk16/NNNNN.npz``: depth ray-cast from the fused TSDF mesh at each keyframe -- consistent
  across the views, unlike a per-frame prediction -- with ``valid`` (a surface was hit) and ``confidence``
  (1 within two subject distances of the camera, 0.3 beyond: Fire3D's 0.6 threshold drops the far field);
- ``camera.json``: per-frame ``intrinsics_K`` and OpenCV ``poses_c2w`` in the export's levelled frame (z up),
  scaled so the cameras stand ~3 m from the subject: Fire3D's priors are room-sized, our SfM units arbitrary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)

TARGET_DISTANCE_M = 3.0  # camera-to-subject distance the scene is scaled to (a room's, as Fire3D was trained on)


def export_scene(run: Path, k: int, out: Path, *, width: int = 960) -> Path:
    import open3d as o3d
    import open3d.core as o3c
    import pycolmap
    from PIL import Image

    from drone3d.export.stage import _subject

    sfm = json.loads((run / "sfm" / "result.json").read_text())
    model_dir = Path(sfm["models"][k]["path"])
    name = model_dir.name
    frame = json.loads((run / "export" / f"model_{name}" / "frame.json").read_text())
    fs, fr, ft = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])
    rec = pycolmap.Reconstruction(str(model_dir))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    if len(ims) > 300:
        ims = [ims[int(round(i))] for i in np.linspace(0, len(ims) - 1, 300)]
    mesh = o3d.io.read_triangle_mesh(str(run / "dense" / f"model_{name}" / "mesh.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    cams = np.array([im.projection_center() for im in ims])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in ims])
    sub = _subject(cams, axes)
    dist = sub[1] if sub is not None else float(np.median(np.linalg.norm(np.asarray(mesh.vertices)[::50] - cams.mean(0), axis=1)))
    metric = TARGET_DISTANCE_M / (fs * dist)  # export units -> "metres"

    dst = out / "scenes" / f"{run.name}_m{name}"
    for sub_dir in ("rgb", "depth_pi3x_conf_chunk16"):
        (dst / sub_dir).mkdir(parents=True, exist_ok=True)
    poses, ks = [], []
    size = None
    for i, im in enumerate(ims):
        cam = rec.cameras[im.camera_id]
        s = width / cam.width
        w, h = width, round(cam.height * s) // 2 * 2
        size = (w, h)
        f, cx, cy = cam.params[0] * s, cam.params[1] * s, cam.params[2] * s
        pose = im.cam_from_world()
        rcw, tcw = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
        centre = -rcw.T @ tcw
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        d = np.stack([(xs + 0.5 - cx) / f, (ys + 0.5 - cy) / f, np.ones_like(xs)], -1).reshape(-1, 3)
        rays = np.concatenate([np.broadcast_to(centre, d.shape), d @ rcw], 1).astype(np.float32)  # unnormalised: t = z
        z = scene.cast_rays(o3c.Tensor(rays))["t_hit"].numpy().reshape(h, w)
        valid = np.isfinite(z)
        depth = np.where(valid, z * fs * metric, 0.0).astype(np.float32)
        conf = np.where(valid & (depth <= 2.0 * TARGET_DISTANCE_M), 1.0, 0.3).astype(np.float32)
        np.savez_compressed(dst / "depth_pi3x_conf_chunk16" / f"{i:05d}.npz", depth=depth, valid=valid, confidence=conf)
        Image.open(run / "dataset" / "images" / im.name).convert("RGB").resize((w, h), Image.LANCZOS).save(dst / "rgb" / f"{i:05d}.png")
        c2w = np.eye(4)
        c2w[:3, :3] = fr @ rcw.T  # camera axes (OpenCV) in the levelled export frame
        c2w[:3, 3] = (fs * fr @ centre + ft) * metric
        poses.append(c2w.tolist())
        ks.append([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    (dst / "camera.json").write_text(json.dumps({"width": size[0], "height": size[1], "poses_c2w": poses, "intrinsics_K": ks,
                                                 "source": str(run), "model": name, "metres_per_export_unit": metric}))  # fmt: skip
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("model", type=int, nargs="?", default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=960)
    args = ap.parse_args()
    out = args.out or args.run / "fire3d" / "input"
    print(export_scene(args.run, args.model, out, width=args.size))


if __name__ == "__main__":
    main()
