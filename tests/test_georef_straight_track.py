"""Georeferencing must survive a straight single-pass track.

Reproduces the failure the audit found: with camera centres on a line, a
7-DoF similarity cannot fix the roll about the flight axis, and GPS noise
rotates the model so ground away from the track lands tens of metres off,
while the in-sample RMSE still looks like GPS noise.
"""

from __future__ import annotations

import numpy as np

from drone3d.geo.georef import SimilarityTransform, solve_georef, solve_similarity


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def _scene(track: str, seed: int):
    rng = np.random.default_rng(seed)
    if track == "line":
        s = np.linspace(-150, 150, 60)
        cams_world = np.c_[s, 0.3 * np.sin(s / 40), np.full_like(s, 80.0)]
    else:  # lawnmower survey: two parallel legs
        s = np.linspace(-150, 150, 30)
        cams_world = np.r_[np.c_[s, np.full_like(s, -60), np.full_like(s, 80.0)],
                           np.c_[s[::-1], np.full_like(s, 60), np.full_like(s, 80.0)]]  # fmt: skip
    ground = np.c_[
        rng.uniform(-200, 200, 3000), rng.uniform(-150, 150, 3000), rng.normal(0, 0.3, 3000)
    ]
    buildings = np.c_[rng.uniform(-50, 50, 300), rng.uniform(-40, 40, 300), rng.uniform(0, 25, 300)]
    points_world = np.r_[ground, buildings]
    # The SfM model lives in an arbitrary similarity frame.
    to_model = SimilarityTransform(0.2, _random_rotation(rng), rng.normal(0, 5, 3))
    cams_model = to_model.apply(cams_world)
    points_model = to_model.apply(points_world)
    gps = (
        cams_world
        + np.c_[rng.normal(0, 1.5, (len(cams_world), 2)), rng.normal(0, 3.0, len(cams_world))]
    )
    probes_world = np.array([[0.0, 100.0, 0.0], [0.0, -100.0, 0.0], [120.0, 90.0, 0.0]])
    return cams_model, points_model, gps, to_model.apply(probes_world), probes_world


def _probe_error(transform: SimilarityTransform, probes_model, probes_world) -> float:
    return float(np.max(np.linalg.norm(transform.apply(probes_model) - probes_world, axis=1)))


def test_straight_track_uses_ground_levelled_fit_and_stays_accurate() -> None:
    worst_auto, worst_sim = 0.0, 0.0
    for seed in range(8):
        cams, points, gps, probes_model, probes_world = _scene("line", seed)
        transform, info = solve_georef(cams, gps, points, mode="auto")
        assert info["mode"] == "yaw-scale"
        assert info["track"]["linearity"] < 0.15
        worst_auto = max(worst_auto, _probe_error(transform, probes_model, probes_world))
        worst_sim = max(
            worst_sim, _probe_error(solve_similarity(cams, gps), probes_model, probes_world)
        )
    # Ground 100 m off-track stays within a few metres with the levelled fit ...
    assert worst_auto < 5.0
    # ... while the unconstrained similarity is off by tens of metres.
    assert worst_sim > 4 * worst_auto


def test_two_dimensional_track_keeps_full_similarity() -> None:
    cams, points, gps, probes_model, probes_world = _scene("survey", 3)
    transform, info = solve_georef(cams, gps, points, mode="auto")
    assert info["mode"] == "similarity"
    assert _probe_error(transform, probes_model, probes_world) < 5.0


def test_held_out_error_is_reported_and_not_smaller_than_in_sample() -> None:
    cams, points, gps, *_ = _scene("line", 1)
    _, info = solve_georef(cams, gps, points, mode="auto")
    held_out = info["held_out"]["loo_rmse_horizontal_m"]
    in_sample = info["in_sample"]["rmse_horizontal_m"]
    assert held_out >= in_sample
    assert 1.0 < held_out < 4.0  # consistent with 1.5 m per-axis GPS noise
