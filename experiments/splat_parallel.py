"""Splat training throughput: one model at a time against several at once, on the same models.

    uv run python experiments/splat_parallel.py outputs/RUN [1 2 3]   (writes paper/figures/splat_parallel.json)

Trains every splat-eligible model of RUN with the phase-2 settings (low quality, 7000
steps, dense init from the run's sparse_init/) into a scratch directory, for each
concurrency, and reports wall time and mean held-out PSNR. A trainer's kernels are
small; between them one process leaves the GPU idle, which another can fill.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import drone3d  # noqa: F401  (thread binding)
from drone3d.pipeline import _mean_metric
from drone3d.splat.spirula import run_train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    levels = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    dataset = run / "dataset"
    sfm = json.loads((run / "sfm" / "result.json").read_text())
    models = [m for m in sfm["models"] if m["images"] >= 8 and (dataset / "sparse_init" / Path(m["path"]).name).is_dir()]
    scratch = ROOT / "outputs" / "experiments" / "splat_parallel"
    rows = []
    for par in levels:
        out = scratch / f"p{par}"
        shutil.rmtree(out, ignore_errors=True)

        def train(m: dict, out: Path = out) -> dict:
            k = Path(m["path"]).name
            r = run_train(dataset, out / f"model_{k}", recon_dir=f"sparse_init/{k}", iterations=7000, quality="low",
                          depth_weight=0.0, eval_interval=8, flags={"train_resolution_divisor": 2, "cache_images": "disk", "load_depths": 0})  # fmt: skip
            return r.to_dict()

        t0 = time.perf_counter()
        with ThreadPoolExecutor(par) as pool:
            res = list(pool.map(train, models))
        wall = time.perf_counter() - t0
        psnr = [v for v in (_mean_metric(r["eval_metrics"], "psnr") for r in res) if v == v]  # NaN: no eval
        row = {"run": run.name, "parallel": par, "models": len(models), "wall_s": round(wall, 1),
               "sum_model_s": round(sum(r.get("seconds") or 0 for r in res), 1),
               "psnr_mean": round(sum(psnr) / len(psnr), 2) if psnr else None}  # fmt: skip
        rows.append(row)
        print(row, flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    fig = ROOT / "paper" / "figures" / "splat_parallel.json"
    fig.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
