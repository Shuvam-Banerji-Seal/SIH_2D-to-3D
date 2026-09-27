"""TSDF settings against completeness: depth maps once per model, fusion per setting.

For each model of a finished fast run, depth references are computed with the
dense stage's own code (flow triangulation + Depth Anything fill), then fused
with every (voxel size, truncation band) combination; each mesh is scored by
exact view completeness (a ray per pixel against the mesh, non-sky pixels of
the references) and its fusion time.

    uv run python experiments/dense_sweep.py outputs/jal_mahal_rel6 [--models 0,1]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--models", default="")
    ap.add_argument("--voxel-px", default="1.5,2,3")
    ap.add_argument("--trunc", default="4,8,12")
    args = ap.parse_args()

    import torch

    from drone3d.fastsfm.dense import tsdf_fuse
    from drone3d.fastsfm.dense_stage import compute_depths, mesh_depth_error, view_coverage
    from drone3d.fastsfm.mono import MonoDepth
    from drone3d.keyframes.flow import RaftFlow

    sfm = json.loads((args.run / "sfm" / "result.json").read_text())
    models = [Path(m["path"]) for m in sfm["models"] if m["images"] >= 3]
    if args.models:
        keep = set(args.models.split(","))
        models = [m for m in models if m.name in keep]
    raft = RaftFlow("raft_large", batch=16, iters=12)
    mono = MonoDepth("depth-anything/Depth-Anything-V2-Large-hf")
    rows = []
    for model in models:
        r = compute_depths(model, args.run / "dataset" / "images", raft, mono, long_side=480, gaps=(2, 4, 8, 12),
                           keyframe_stride=2, min_angle_deg=0.5, rel_tol=0.05)  # fmt: skip
        depths, cams, sky = r["depths"], r["cams"], r["sky"]
        rgb = r["frames"].cpu().numpy()
        n, h, w, _ = rgb.shape
        valid = np.concatenate([d[d > 0] for d in depths]) if any((d > 0).any() for d in depths) else np.zeros(0)
        if not len(valid):
            continue
        med, dmax = float(np.median(valid)), float(np.percentile(valid, 99.5))
        depth_cov = float(np.mean([((d > 0) & ~s).sum() / max(1, (~s).sum()) for d, s in zip(depths, sky, strict=True)]))
        for vpx, trunc in itertools.product(map(float, args.voxel_px.split(",")), map(float, args.trunc.split(","))):
            t0 = time.perf_counter()
            vbg, voxel = tsdf_fuse([(depths[i], rgb[i], cams[i]) for i in range(n)], voxel=vpx * med / cams[0].f,
                                   depth_max=dmax, trunc_voxels=trunc)  # fmt: skip
            mesh = vbg.extract_triangle_mesh().to_legacy()
            dt = time.perf_counter() - t0
            del vbg
            torch.cuda.empty_cache()
            vs, ts = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            comp = view_coverage(vs, ts, cams, (w, h), sky)
            err = mesh_depth_error(vs, ts, cams, (w, h), r["tri_depths"])
            row = {"model": model.name, "refs": n, "depth_cov": round(depth_cov, 3), "voxel_px": vpx, "trunc": trunc,
                   "completeness": round(comp, 3), "depth_err_vs_triangulated": round(err, 4),
                   "triangles": len(ts), "fuse_s": round(dt, 2)}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.run / "dense_sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
