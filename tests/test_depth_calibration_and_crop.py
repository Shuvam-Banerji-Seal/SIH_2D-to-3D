"""Monotone depth calibration and letterbox detection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from drone3d.depth.align import MonotoneMap, _pav, cross_validate


def test_pav_is_non_decreasing_and_weight_preserving() -> None:
    y = np.array([1.0, 3.0, 2.0, 2.0, 5.0, 4.0])
    w = np.ones_like(y)
    out = _pav(y, w)
    assert np.all(np.diff(out) >= 0)
    assert out.sum() == pytest.approx(y.sum())  # pooling keeps the weighted mean


def test_monotone_beats_affine_when_far_field_is_compressed() -> None:
    # A prior that compresses depth beyond ~20 m (log depth concave in pred),
    # as Marigold v2 does at landscape scale.
    rng = np.random.default_rng(0)
    z = np.exp(rng.uniform(np.log(5.0), np.log(500.0), 6000))
    lz = np.log(z)
    pred = np.where(lz < 3.0, lz, 3.0 + 0.25 * (lz - 3.0)) + rng.normal(0, 0.01, z.shape)
    cv = cross_validate(pred, z)
    assert cv is not None
    assert cv["monotone"]["abs_rel_median"] < 0.5 * cv["affine"]["abs_rel_median"]
    assert cv["monotone"]["delta1"] > cv["affine"]["delta1"]


def test_monotone_map_extrapolates_with_the_affine_slope() -> None:
    p = np.linspace(-0.5, 0.5, 400)
    z = np.exp(2.0 * p + 1.0)
    m = MonotoneMap(p, z, slope=2.0)
    assert m(np.array([0.0]))[0] == pytest.approx(1.0, abs=0.02)
    assert m(np.array([1.0]))[0] == pytest.approx(3.0, abs=0.05)  # beyond the knots


def test_scaled_crop_rounds_inwards_to_multiples() -> None:
    from drone3d.io.nvdec import scaled_crop

    x0, y0, x1, y1 = scaled_crop((0, 114, 3840, 2046), (3840, 2160), (640, 360), multiple=8)
    assert (x1 - x0) % 8 == 0 and (y1 - y0) % 8 == 0
    assert y0 >= 114 * 360 / 2160 and y1 <= 2046 * 360 / 2160


@pytest.mark.parametrize("bars", [48, 0])
def test_detect_letterbox_on_synthetic_clip(tmp_path: Path, bars: int) -> None:
    from drone3d.io.nvdec import detect_letterbox, ffmpeg_bin, probe_stream

    clip = tmp_path / "clip.mp4"
    content_h = 360 - 2 * bars
    vf = f"pad=640:360:0:{bars}:black" if bars else "null"
    subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=640x{content_h}:rate=10", "-vf", vf, "-t", "4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    crop = detect_letterbox(probe_stream(clip), hwaccel=False, samples=6)
    if bars == 0:
        assert crop is None
    else:
        assert crop is not None
        x0, y0, x1, y1 = crop
        assert (x0, x1) == (0, 640)
        assert bars <= y0 <= bars + 4 and 360 - bars - 4 <= y1 <= 360 - bars


def test_aggregate_reports_every_model_including_all_failed() -> None:
    from drone3d.depth.stage import aggregate

    cv = {
        "affine": {"abs_rel_median": 0.05, "delta1": 0.9},
        "monotone": {"abs_rel_median": 0.03, "delta1": 0.95},
    }
    ok = {
        "model": "m0",
        "abs_rel": 0.06,
        "abs_rel_median": 0.05,
        "delta1": 0.9,
        "valid_fraction": 1.0,
        "cv": cv,
    }
    per_image = {
        "a.jpg": ok,
        "b.jpg": dict(ok),
        "c.jpg": {"model": "m1", "status": "alignment-failed"},
    }
    out = aggregate(per_image)
    assert set(out) == {"m0", "m1"}
    assert out["m0"]["aligned"] == 2 and out["m0"]["cv_monotone_abs_rel_median"] == 0.03
    assert out["m1"] == {
        **out["m1"],
        "images": 1,
        "aligned": 0,
        "abs_rel_median": None,
        "cv_monotone_delta1": None,
    }
