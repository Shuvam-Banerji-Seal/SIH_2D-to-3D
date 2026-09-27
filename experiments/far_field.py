"""Where the fused depth is empty, and how well the monocular fill extrapolates beyond the calibrated range.

    uv run python experiments/far_field.py outputs/RUN [model indices...]   (writes paper/figures/far_field.json)

1. Of every keyframe's pixels: sky (the prior's near-zero disparity), triangulated,
   filled, and left empty -- split into beyond the far cut (3x the 99th-percentile
   triangulated depth) and otherwise (no valid prediction / calibration refused).
2. Held-out extrapolation: calibrate each keyframe on its nearer 80 % of
   triangulated pixels only, predict the farthest 20 %, and report the median
   relative error by how far past the calibrated range (its 99th percentile) a
   pixel lies. This is what raising the far cut would put into the mesh.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)
from drone3d.engine import models
from drone3d.fastsfm.dense_stage import compute_depths
from drone3d.fastsfm.mono import calibrate_fill

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0.0, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99.0)]


def main() -> None:
    run = Path(sys.argv[1])
    picks = [int(x) for x in sys.argv[2:]] or [0]
    raft = models.raft("raft_large", batch=16, iters=12)
    mono = models.mono()
    rows = []
    for k in picks:
        model = run / "dataset" / "sparse" / str(k)
        if not model.is_dir():
            continue
        r = compute_depths(model, run / "dataset" / "images", raft, mono, long_side=480, gaps=(2, 4, 8, 12),
                           keyframe_stride=1, min_angle_deg=0.5, rel_tol=0.05)  # fmt: skip
        disp = mono(r["frames"]).cpu().numpy()
        share = {"sky": [], "triangulated": [], "filled": [], "empty_far": [], "empty_other": []}
        errs: list[list[float]] = [[] for _ in BINS]
        for i, (tri, fused) in enumerate(zip(r["tri_depths"], r["depths"], strict=True)):
            sky = r["sky"][i]
            have = (tri > 0) & ~sky
            share["sky"].append(sky.mean())
            share["triangulated"].append(have.mean())
            share["filled"].append(((fused > 0) & ~have).mean())
            empty = (fused <= 0) & ~sky
            if have.sum() >= 400:
                full, _ = calibrate_fill(disp[i], tri, far_factor=1e9)  # no far cut: where would the prior put them?
                far = 3.0 * float(np.quantile(tri[have], 0.99))
                share["empty_far"].append((empty & (full > far)).mean())
                share["empty_other"].append((empty & ~(full > far)).mean())
            else:
                share["empty_far"].append(0.0)
                share["empty_other"].append(empty.mean())
            # held-out extrapolation
            if have.sum() < 2000:
                continue
            z = tri[have]
            cut = float(np.quantile(z, 0.8))
            near = np.where(have & (tri <= cut), tri, 0.0).astype(np.float32)
            pred, info = calibrate_fill(disp[i], near, far_factor=1e9)
            if info.get("status") != "filled":
                continue
            ref_max = float(np.quantile(tri[near > 0], 0.99))
            test = have & (tri > cut) & (pred > 0)
            ratio = tri[test] / ref_max
            rel = np.abs(pred[test] - tri[test]) / tri[test]
            for b, (lo, hi) in enumerate(BINS):
                sel = (ratio > lo) & (ratio <= hi)
                if sel.sum() >= 50:
                    errs[b].append(float(np.median(rel[sel])))
        row = {"run": run.name, "model": k, "keyframes": len(r["depths"]),
               "shares": {key: round(float(np.mean(v)), 4) for key, v in share.items()},
               "extrapolation": [{"ratio": f"{lo:g}-{hi:g}", "images": len(e), "median_rel_err": round(float(np.median(e)), 4) if e else None}
                                 for (lo, hi), e in zip(BINS, errs, strict=True)]}  # fmt: skip
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = ROOT / "paper" / "figures" / "far_field.json"
    old = json.loads(out.read_text()) if out.is_file() else []
    out.write_text(json.dumps([x for x in old if x["run"] != run.name] + rows, indent=1))


if __name__ == "__main__":
    main()
