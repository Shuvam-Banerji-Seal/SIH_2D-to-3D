"""Real-run footage for the promo film's screen slots, from a fast-profile run.

* ``depth_tiles.mp4`` -- 2 x 2 tiles per keyframe: the keyframe, the depth
  triangulated from flow, the depth completed by the calibrated prior, and the
  textured mesh rendered from the same pose;
* ``model_inset.mp4`` and ``flythrough.mp4`` -- the textured mesh flown along
  the drone's own path (ray-cast, ``drone3d.export.render_mesh``).

    uv run python promo/assets.py outputs/<run> [model index]
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


def main() -> None:
    from drone3d.export.render_mesh import render_mesh_flythrough

    run = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    BUILD.mkdir(parents=True, exist_ok=True)
    sfm = json.loads((run / "sfm" / "result.json").read_text())
    model = Path(sfm["models"][k]["path"])
    export_dir = run / "export" / f"model_{model.name}"
    print("depth tiles ->", depth_tiles(run, model, export_dir, BUILD / "depth_tiles.mp4"), flush=True)
    print("inset ->", render_mesh_flythrough(export_dir, model, BUILD / "model_inset.mp4", seconds=6, size=(760, 428)), flush=True)
    print("fly-through ->", render_mesh_flythrough(export_dir, model, BUILD / "flythrough.mp4", seconds=10, size=(1920, 1080)), flush=True)


if __name__ == "__main__":
    main()
