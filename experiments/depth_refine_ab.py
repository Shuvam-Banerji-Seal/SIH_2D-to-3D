"""A/B of edge-aware depth refinement (OpenCV ximgproc) on real models: completeness, depth error, time.

    uv run python experiments/depth_refine_ab.py outputs/RUN [model indices...]  (writes paper/figures/depth_refine.json)

For each model: the fast profile's fused depth maps, then none / guided / fgs
refinement, the same TSDF, and the exact view completeness and the median
depth error of the mesh against the triangulated (measured) depth.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)
from drone3d.engine import models
from drone3d.fastsfm.dense import tsdf_fuse
from drone3d.fastsfm.dense_stage import compute_depths, mesh_depth_error, view_coverage
from drone3d.fastsfm.refine import refine_depths

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run = Path(sys.argv[1])
    picks = [int(x) for x in sys.argv[2:]] or [0, 1]
    raft = models.raft("raft_large", batch=16, iters=12)
    mono = models.mono()
    rows = []
    for k in picks:
        model = run / "dataset" / "sparse" / str(k)
        if not model.is_dir():
            continue
        r = compute_depths(model, run / "dataset" / "images", raft, mono, long_side=480, gaps=(2, 4, 8, 12),
                           keyframe_stride=1, min_angle_deg=0.5, rel_tol=0.05)  # fmt: skip
        cams, frames, sky = r["cams"], r["frames"], r["sky"]
        n, h, w, _ = frames.shape
        rgb = frames.cpu().numpy()
        valid = np.concatenate([d[d > 0] for d in r["depths"]])
        voxel = 3.0 * float(np.median(valid)) / cams[0].f
        for method in ("none", "guided", "fgs"):
            t0 = time.perf_counter()
            depths = refine_depths(r["depths"], rgb, r["tri_depths"], method=method)
            t_ref = time.perf_counter() - t0
            vbg, vx = tsdf_fuse([(depths[i], rgb[i], cams[i]) for i in range(n)], voxel=voxel,
                                depth_max=float(np.percentile(valid, 99.5)), trunc_voxels=12.0, memory_gb=8.0)  # fmt: skip
            mesh = vbg.extract_triangle_mesh().to_legacy()
            v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            row = {"run": run.name, "model": k, "keyframes": n, "method": method, "refine_s": round(t_ref, 2),
                   "completeness": round(view_coverage(v, t, cams, (w, h), sky), 4),
                   "depth_err": round(mesh_depth_error(v, t, cams, (w, h), r["tri_depths"]), 5), "triangles": len(t)}  # fmt: skip
            rows.append(row)
            print(row, flush=True)
            del vbg, mesh
    out = ROOT / "paper" / "figures" / "depth_refine.json"
    old = json.loads(out.read_text()) if out.is_file() else []
    keep = [x for x in old if x["run"] != run.name]
    out.write_text(json.dumps(keep + rows, indent=1))


if __name__ == "__main__":
    main()
