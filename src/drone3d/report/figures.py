"""Diagnostic figures for the report and the paper (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["keyframe_timeline"]

_VERDICT_COLOURS = {
    "3d": "#2e7d32",
    "weak-3d": "#f9a825",
    "degenerate": "#c62828",
    "too-short": "#757575",
}


def keyframe_timeline(selection: dict, out_path: str | Path, fps: float | None = None) -> Path:
    """Consistency, passes, keyframes and verdicts along the video's timeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.asarray(selection["analysis_frame_indices"], dtype=float)
    cons = np.asarray(selection["consistency"], dtype=float)
    sharp = np.asarray(selection["sharpness"], dtype=float)
    passes = selection["passes"]
    if fps is None:
        # timestamps of keyframes give frame index -> seconds
        k = selection["keyframes"]
        fps = k[0]["frame_index"] / k[0]["timestamp_s"] if k and k[0]["timestamp_s"] > 0 else 30.0
        if len(k) > 1 and k[-1]["timestamp_s"] > 0:
            fps = k[-1]["frame_index"] / k[-1]["timestamp_s"]
    t = idx / fps
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1]}
    )
    ax = axes[0]
    ax.plot(t[1:], cons, color="#1565c0", lw=1.0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("flow consistency\n(frame to next)")
    for p in passes:
        colour = _VERDICT_COLOURS.get(p["verdict"], "#9e9e9e")
        for a in axes:
            a.axvspan(p["start_s"], p["end_s"], color=colour, alpha=0.10, lw=0)
        snr = p.get("parallax_snr")
        label = f"pass {p['pass_id']}: {p['verdict']}" + (
            f"\nSNR {snr:.1f}" if snr is not None else ""
        )
        ax.text(
            (p["start_s"] + p["end_s"]) / 2,
            0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=colour,
        )
    ax.set_title("Pass segmentation, keyframes and 3D verdict")

    ax = axes[1]
    kf_t = np.array([k["timestamp_s"] for k in selection["keyframes"]])
    kf_ov = np.array(
        [np.nan if k["overlap_prev"] is None else k["overlap_prev"] for k in selection["keyframes"]]
    )
    ax.scatter(kf_t, kf_ov, s=9, color="#6a1b9a", label="overlap with previous keyframe")
    target = selection.get("config", {}).get("overlap_target")
    band = selection.get("config", {}).get("overlap_band")
    if target is not None and band is not None:
        ax.axhspan(target, target + band, color="#6a1b9a", alpha=0.12, lw=0, label="selection band")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("overlap")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[2]
    rel = sharp / np.median(sharp[sharp > 0]) if np.any(sharp > 0) else sharp
    ax.plot(t, rel, color="#546e7a", lw=0.8, label="relative sharpness")
    ax.vlines(
        kf_t, 0, rel.max() if len(rel) else 1, color="#6a1b9a", lw=0.4, alpha=0.6, label="keyframes"
    )
    ax.set_ylabel("sharpness")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
