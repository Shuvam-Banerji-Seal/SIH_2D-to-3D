"""Chained vs direct optical-flow correspondence error, against exact ground truth.

A pure-rotation clip (``tools/make_degenerate_clip.py``) has an exact inter-frame
homography ``x_j = K R_j R_i^T K^-1 x_i``. This measures, for growing frame
gaps, the error of (a) chaining consecutive RAFT flows through the keyframe
tracker and (b) running RAFT directly between the two frames. Chained error
grows about linearly with the gap -- a systematic per-step bias, not random
noise -- which is why the 3D verdict uses direct flow.

    uv run python experiments/flow_drift.py outputs/controls/pure_rotation.mp4 paper/figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)

sys.path.insert(0, str(ROOT / "tools"))

from make_degenerate_clip import rotation  # noqa: E402

from drone3d.io.nvdec import analysis_size, probe_stream, stream_analysis_chunks  # noqa: E402
from drone3d.keyframes.flow import ConsecutiveFlow, RaftFlow, consistency_mask  # noqa: E402
from drone3d.keyframes.select import Tracker  # noqa: E402

# Must match the generator's defaults.
SECONDS, FPS, YAW, OUT_W = 8.0, 30, 24.0, 1920
GAPS = (1, 2, 4, 8, 12, 16, 24, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = probe_stream(args.clip)
    size = analysis_size(info.width, info.height, 640)
    stride = 2
    flow_model = RaftFlow("raft_large", batch=32)
    flow, _, _ = ConsecutiveFlow.compute(
        stream_analysis_chunks(info, size, stride=stride), flow_model
    )
    frames = flow.frames
    n_src = int(SECONDS * FPS)
    s = size[0] / OUT_W
    f = 0.5 * OUT_W / math.tan(math.radians(22)) * s
    k = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])

    def rot(a: int) -> np.ndarray:  # analysis frame -> source-frame rotation
        u = (a * stride) / (n_src - 1)
        return rotation(YAW * (u - 0.5), 3.0 * math.sin(2 * math.pi * u), 2.0 * (u - 0.5))

    def truth(a: int, b: int, x: np.ndarray) -> np.ndarray:
        h = k @ rot(b).T @ rot(a) @ np.linalg.inv(k)
        y = np.c_[x, np.ones(len(x))] @ h.T
        return y[:, :2] / y[:, 2:]

    tracker = Tracker(flow, 8, 32)
    grid = tracker.grid.cpu().numpy()
    starts = list(range(8, flow.num_frames - max(GAPS) - 1, 10))
    rows = []
    for gap in GAPS:
        chained, direct = [], []
        for a in starts:
            b = a + gap
            pts, alive = tracker.fresh()
            for t in range(a, b):
                pts, alive = tracker.step(pts, alive, t)
            m = alive[0].cpu().numpy()
            chained.append(np.linalg.norm(pts[0].cpu().numpy()[m] - truth(a, b, grid[m]), axis=1))
            fwd = flow_model(frames[a : a + 1], frames[b : b + 1])
            bwd = flow_model(frames[b : b + 1], frames[a : a + 1])
            ok = consistency_mask(fwd, bwd)[0]
            xs, ys = grid[:, 0].astype(int), grid[:, 1].astype(int)
            keep = ok[ys, xs].cpu().numpy()
            disp = fwd[0, :, ys, xs].T.cpu().numpy()
            direct.append(np.linalg.norm(grid[keep] + disp[keep] - truth(a, b, grid[keep]), axis=1))
        c, d = np.concatenate(chained), np.concatenate(direct)
        rows.append(
            {
                "gap_frames": gap,
                "chained_median_px": float(np.median(c)),
                "chained_p90_px": float(np.quantile(c, 0.9)),
                "direct_median_px": float(np.median(d)),
                "direct_p90_px": float(np.quantile(d, 0.9)),
                "pairs": len(starts),
            }
        )
        print(json.dumps(rows[-1]))
    (out / "flow_drift.json").write_text(json.dumps(rows, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [r["gap_frames"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(
        g,
        [r["chained_median_px"] for r in rows],
        "o-",
        color="#c62828",
        label="chained consecutive flow",
    )
    ax.fill_between(
        g,
        [r["chained_median_px"] for r in rows],
        [r["chained_p90_px"] for r in rows],
        color="#c62828",
        alpha=0.12,
    )
    ax.plot(
        g,
        [r["direct_median_px"] for r in rows],
        "s-",
        color="#1565c0",
        label="direct pairwise flow",
    )
    ax.fill_between(
        g,
        [r["direct_median_px"] for r in rows],
        [r["direct_p90_px"] for r in rows],
        color="#1565c0",
        alpha=0.12,
    )
    ax.set_xlabel("frame gap (analysis frames, 15 fps)")
    ax.set_ylabel("correspondence error (px @ 640 px)")
    ax.set_title("RAFT correspondence error vs ground truth", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "flow_drift.pdf")
    fig.savefig(out / "flow_drift.png", dpi=160)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
