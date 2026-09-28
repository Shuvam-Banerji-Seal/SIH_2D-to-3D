"""Keyframes and their flow-triangulated depth, as the reference for comparing depth priors (experiments/prior_bench.py).

    uv run python experiments/prior_dump.py RUN:MODEL [RUN:MODEL ...]   (writes outputs/experiments/prior_ref.npz)

For each SfM model: up to 12 evenly spaced keyframes at the dense stage's 480 px, their triangulated depth (only
the flow triangulation -- no prior), the sky mask of the current prior, and the image paths at full resolution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)
from drone3d.engine import models
from drone3d.fastsfm.dense_stage import compute_depths

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raft, mono = models.raft("raft_large", batch=16, iters=12), models.mono()
    out: dict[str, np.ndarray] = {}
    index = []
    for arg in sys.argv[1:]:
        run, k = arg.rsplit(":", 1)
        run_dir = ROOT / "outputs" / run
        r = compute_depths(run_dir / "dataset" / "sparse" / k, run_dir / "dataset" / "images", raft, mono, long_side=480,
                           gaps=(2, 4, 8, 12), keyframe_stride=1, min_angle_deg=0.5, rel_tol=0.05)  # fmt: skip
        tri, sky, ims = r["tri_depths"], r["sky"], r["ims"]
        pick = [i for i in np.linspace(0, len(tri) - 1, min(12, len(tri))).round().astype(int) if (tri[i] > 0).sum() > 2000]
        for i in pick:
            key = f"{run}:{k}:{ims[i].name}"
            out[f"tri/{key}"] = tri[i].astype(np.float32)
            out[f"sky/{key}"] = sky[i]
            index.append({"key": key, "run": run, "model": k, "image": str(run_dir / "dataset" / "images" / ims[i].name),
                          "size": list(tri[i].shape[::-1]), "triangulated": round(float((tri[i] > 0).mean()), 4)})  # fmt: skip
        print(run, k, len(pick), "keyframes", flush=True)
    dst = ROOT / "outputs" / "experiments" / "prior_ref.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    (dst.with_suffix(".json")).write_text(json.dumps(index, indent=1))
    print(len(index), "keyframes ->", dst)


if __name__ == "__main__":
    main()
