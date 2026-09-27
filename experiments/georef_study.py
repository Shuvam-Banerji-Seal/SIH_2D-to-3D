"""Georeferencing a single-pass track: 7-DoF similarity vs ground-levelled 4-DoF fit.

Synthetic scenes with known truth: a camera track at 80 m altitude over a
ground plane with buildings, the SfM model hidden behind a random similarity,
consumer-grade GPS noise. Sweeps horizontal GPS noise and the track's lateral
deviation (0 m = a perfectly straight pass) and records the error of ground
points 100 m off-track, the in-sample RMSE, and the leave-one-out RMSE.

    uv run python experiments/georef_study.py paper/figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone3d.geo.georef import SimilarityTransform, solve_georef, solve_similarity  # noqa: E402

TRIALS = 20


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def trial(noise_h: float, lateral: float, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    s = np.linspace(-150, 150, 60)
    cams = np.c_[s, lateral * np.sin(np.pi * s / 150), np.full_like(s, 80.0)]
    ground = np.c_[
        rng.uniform(-200, 200, 3000), rng.uniform(-150, 150, 3000), rng.normal(0, 0.3, 3000)
    ]
    blds = np.c_[rng.uniform(-50, 50, 300), rng.uniform(-40, 40, 300), rng.uniform(0, 25, 300)]
    to_model = SimilarityTransform(rng.uniform(0.05, 5.0), _rotation(rng), rng.normal(0, 5, 3))
    gps = cams + np.c_[rng.normal(0, noise_h, (60, 2)), rng.normal(0, 2 * noise_h, 60)]
    probes = np.c_[rng.uniform(-150, 150, 50), rng.choice([-100.0, 100.0], 50), np.zeros(50)]
    cm, pm, prm = to_model.apply(cams), to_model.apply(np.r_[ground, blds]), to_model.apply(probes)
    auto, info = solve_georef(cm, gps, pm, mode="auto")
    sim = solve_similarity(cm, gps)

    def probe_rmse(t: SimilarityTransform) -> float:
        return float(np.sqrt(np.mean(np.sum((t.apply(prm) - probes) ** 2, axis=1))))

    return {
        "auto_mode": info["mode"],
        "auto_probe_rmse_m": probe_rmse(auto),
        "sim_probe_rmse_m": probe_rmse(sim),
        "auto_in_sample_h_m": info["in_sample"]["rmse_horizontal_m"],
        "auto_loo_h_m": info["held_out"]["loo_rmse_horizontal_m"],
        "sim_in_sample_h_m": float(
            np.sqrt(np.mean(np.sum((sim.apply(cm) - gps)[:, :2] ** 2, axis=1)))
        ),
        "scale_rel_err": abs(auto.scale * to_model.scale - 1.0),
        "scale_rel_std": info["held_out"]["scale_relative_std"],
    }


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for noise in (0.5, 1.5, 3.0, 5.0):
        for lateral in (0.0, 5.0, 20.0, 60.0):
            trials = [
                trial(noise, lateral, 1000 * int(noise * 10) + 100 * int(lateral) + i)
                for i in range(TRIALS)
            ]
            row = {"gps_noise_h_m": noise, "lateral_m": lateral}
            for key in trials[0]:
                if key == "auto_mode":
                    row["auto_yaw_scale_share"] = float(
                        np.mean([t[key] == "yaw-scale" for t in trials])
                    )
                else:
                    row[key] = float(np.median([t[key] for t in trials]))
            rows.append(row)
            print(
                json.dumps(
                    {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                )
            )
    (out / "georef_study.json").write_text(json.dumps(rows, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=False)
    for ax, lateral in zip(axes, (0.0, 60.0), strict=True):
        sub = [r for r in rows if r["lateral_m"] == lateral]
        x = [r["gps_noise_h_m"] for r in sub]
        ax.plot(
            x, [r["sim_probe_rmse_m"] for r in sub], "o-", color="#c62828", label="7-DoF similarity"
        )
        ax.plot(
            x, [r["auto_probe_rmse_m"] for r in sub], "s-", color="#2e7d32", label="auto (ours)"
        )
        ax.plot(
            x,
            [r["sim_in_sample_h_m"] for r in sub],
            ":",
            color="#c62828",
            alpha=0.7,
            label="similarity in-sample RMSE",
        )
        ax.set_yscale("log")
        ax.set_xlabel("horizontal GPS noise (m, 1σ)")
        ax.set_title(
            "straight pass" if lateral == 0 else f"curved pass (±{lateral:.0f} m)", fontsize=10
        )
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("error 100 m off-track (m, median)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "georef_study.pdf")
    fig.savefig(out / "georef_study.png", dpi=160)


if __name__ == "__main__":
    main()
