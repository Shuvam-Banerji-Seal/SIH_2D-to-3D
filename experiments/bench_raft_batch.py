"""RAFT large throughput by batch size on an idle GPU: does a bigger batch use the GPU better?

    uv run python experiments/bench_raft_batch.py   (writes paper/figures/raft_batch.json)

256 pairs of random frames per size, two timed passes after the CUDA graph capture.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import drone3d  # noqa: F401  (thread binding)
from drone3d.keyframes.flow import RaftFlow, load_raft

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    net = load_raft("raft_large", "cuda")
    rows = []
    for h, w in [(272, 480), (360, 640)]:
        pairs = 256
        a = torch.randint(0, 255, (pairs, h, w, 3), dtype=torch.uint8, device="cuda")
        b = torch.roll(a, 3, dims=2)
        for batch in (16, 32, 64, 128):
            try:
                f = RaftFlow("raft_large", batch=batch, iters=12, net=net)
                f(a[:batch], b[:batch])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(2):
                    f(a, b)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 2
                row = {"size": [w, h], "batch": batch, "flows_per_s": round(pairs / dt, 1),
                       "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1)}  # fmt: skip
            except RuntimeError as exc:  # cuDNN's grid sampler refuses very large batches
                row = {"size": [w, h], "batch": batch, "error": str(exc)[:120]}
            rows.append(row)
            print(row, flush=True)
            f = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    (ROOT / "paper" / "figures" / "raft_batch.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
