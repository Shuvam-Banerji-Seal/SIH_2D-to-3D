"""View completeness of fast-profile runs: how much of what the camera saw was reconstructed.

For every dense model of each run, the mesh is projected into all of that model's
registered keyframes (480 px) and compared with their non-sky pixels (Depth
Anything V2: sky is zero disparity), the same measure the dense stage now reports
for its depth references. Reported per run: keyframes, registered, the
keyframe-weighted mean completeness, triangles, and the processing time.

    uv run python experiments/eval_completeness.py outputs/jal_mahal_fast2 outputs/jal_mahal_rel12 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)


def main() -> None:
    import open3d as o3d
    import pycolmap

    from drone3d.fastsfm.dense import Camera
    from drone3d.fastsfm.dense_stage import view_coverage
    from drone3d.fastsfm.mono import MonoDepth
    from drone3d.fastsfm.stage import load_frames

    mono = MonoDepth("depth-anything/Depth-Anything-V2-Large-hf")
    out = {}
    for run in map(Path, sys.argv[1:]):
        dense = json.loads((run / "dense" / "result.json").read_text())
        sfm = json.loads((run / "sfm" / "result.json").read_text())
        proc = (json.loads((run / "metrics" / "metrics.json").read_text()).get("processing") or {})
        weighted, views, tris = 0.0, 0, 0
        for m in dense["models"]:
            if m.get("status") != "ok":
                continue
            rec = pycolmap.Reconstruction(m["model"])
            ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
            frames, full = load_frames([run / "dataset" / "images" / im.name for im in ims], 480)
            s = frames.shape[2] / full[0]
            cams = []
            for im in ims:
                c = Camera.from_colmap(im, rec.cameras[im.camera_id])
                cams.append(Camera(c.f * s, c.cx * s, c.cy * s, c.k1, c.rotation, c.translation))
            disp = mono(frames).cpu().numpy()
            sky = disp <= 0.005 * np.maximum(disp.reshape(len(disp), -1).max(1), 1e-6)[:, None, None]
            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            share = view_coverage(np.asarray(mesh.vertices), cams, (frames.shape[2], frames.shape[1]), sky)
            weighted += share * len(ims)
            views += len(ims)
            tris += len(mesh.triangles)
        out[run.name] = {
            "keyframes": sfm["input_images"],
            "registered": sfm["registered_images"],
            "view_completeness": round(weighted / views, 4) if views else None,
            "triangles": tris,
            "processing_s": proc.get("seconds"),
            "budget_s": proc.get("budget_seconds"),
        }
        print(run.name, out[run.name], flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
