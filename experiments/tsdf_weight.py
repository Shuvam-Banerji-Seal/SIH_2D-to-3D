"""TSDF extraction: how many depth maps must see a voxel before it becomes surface.

    uv run python experiments/tsdf_weight.py outputs/RUN [model indices...]   (writes paper/figures/tsdf_weight.json)

Open3D extracts a voxel only when its integration weight -- the number of depth
maps that saw it -- is greater than ``weight_threshold`` (default 3, i.e. four
views). That removes floaters but also every surface fewer keyframes saw: the
edges of each model's coverage, and small models entirely. Same depth maps, same
grid, at least 2 / 3 / 4 views: view completeness, the mesh's median depth error
against the triangulated depth, triangles, and the mean number of surfaces a
keyframe's ray crosses where it hits (stacked layers raise it above one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)
from drone3d.engine import models
from drone3d.fastsfm.dense import tsdf_fuse
from drone3d.fastsfm.dense_stage import compute_depths, mesh_depth_error, view_coverage

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
                           keyframe_stride=2, min_angle_deg=0.5, rel_tol=0.05)  # fmt: skip  (stride 2 from 40 views, as the profile)
        cams, frames, sky = r["cams"], r["frames"], r["sky"]
        n, h, w, _ = frames.shape
        rgb = frames.cpu().numpy()
        have = [d[d > 0] for d in r["depths"] if (d > 0).any()]
        if not have:
            rows.append({"run": run.name, "model": k, "keyframes": n, "min_views": None, "status": "no-depth"})
            continue
        valid = np.concatenate(have)
        voxel = 3.0 * float(np.median(valid)) / cams[0].f
        vbg, _ = tsdf_fuse([(r["depths"][i], rgb[i], cams[i]) for i in range(n)], voxel=voxel,
                           depth_max=float(np.percentile(valid, 99.5)), trunc_voxels=12.0, memory_gb=8.0)  # fmt: skip
        for views in (2, 3, 4):
            wt = views - 0.5
            mesh = vbg.extract_triangle_mesh(weight_threshold=wt).to_legacy()
            v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            layers = _layers(v, t, cams, (w, h)) if len(t) else None
            row = {"run": run.name, "model": k, "keyframes": n, "min_views": views, "triangles": len(t),
                   "completeness": round(view_coverage(v, t, cams, (w, h), sky), 4) if len(t) else 0.0,
                   "depth_err": round(mesh_depth_error(v, t, cams, (w, h), r["tri_depths"]), 5) if len(t) else None,
                   "surfaces_per_ray": round(layers, 3) if layers else None}  # fmt: skip
            rows.append(row)
            print(row, flush=True)
        del vbg
    out = ROOT / "paper" / "figures" / "tsdf_weight.json"
    old = json.loads(out.read_text()) if out.is_file() else []
    out.write_text(json.dumps([x for x in old if x["run"] != run.name] + rows, indent=1))


def _layers(v: np.ndarray, t: np.ndarray, cams: list, size: tuple[int, int]) -> float:
    """Mean surfaces crossed by the rays of six keyframes that hit the mesh at all."""
    import open3d as o3d
    import open3d.core as o3c

    w, h = size
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(np.ascontiguousarray(v, dtype=np.float32)), o3c.Tensor(np.ascontiguousarray(t, dtype=np.uint32)))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for c in cams[:: max(1, len(cams) // 6)]:
        d = np.stack([(xs + 0.5 - c.cx) / c.f, (ys + 0.5 - c.cy) / c.f, np.ones_like(xs)], -1).reshape(-1, 3) @ np.asarray(c.rotation, np.float32)
        o = np.broadcast_to(np.asarray(c.centre, np.float32), d.shape)
        n = scene.count_intersections(o3c.Tensor(np.ascontiguousarray(np.concatenate([o, d], 1), dtype=np.float32))).numpy()
        if (n > 0).any():
            out.append(float(n[n > 0].mean()))
    return float(np.mean(out)) if out else 0.0


if __name__ == "__main__":
    main()
