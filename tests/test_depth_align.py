"""Monocular-depth alignment must recover log-affine parameters and sample pixels exactly."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drone3d.depth.align import fit_log_affine, sample_at, write_depth_png  # noqa: E402


def test_recovers_log_affine_despite_outliers() -> None:
    rng = np.random.default_rng(0)
    z = rng.uniform(5.0, 300.0, 4000)
    a_true, b_true = 2.7, 3.1
    pred = (np.log(z) - b_true) / a_true + rng.normal(0, 0.01, z.shape)
    bad = rng.random(z.shape) < 0.2
    pred[bad] = rng.uniform(-1, 1, bad.sum())  # 20 % garbage predictions
    fit = fit_log_affine(pred, z)
    assert fit is not None
    assert fit.a == pytest.approx(a_true, rel=0.02)
    assert fit.b == pytest.approx(b_true, abs=0.05)
    assert 0.75 < fit.inlier_fraction < 0.85
    # AbsRel over ALL points is dominated by the outliers; the median is not.
    assert fit.abs_rel_median < 0.05


def test_degenerate_input_returns_none() -> None:
    assert fit_log_affine(np.zeros(100), np.full(100, 10.0)) is None
    assert fit_log_affine(np.array([0.1, 0.2]), np.array([1.0, 2.0])) is None


def test_sample_at_uses_colmap_pixel_centres() -> None:
    # A 4x8 field whose value is its column index: sampling the centre of
    # column k (COLMAP u = k + 0.5) on a 16-px-wide image at half resolution.
    w, h = 8, 4
    field = torch.arange(w, dtype=torch.float32).repeat(h, 1)[None]
    width, height = 16, 8
    uv = np.array([[2 * k + 1.0, 4.0] for k in range(w)])  # centres of 2x2 blocks
    got = sample_at(field, uv, width, height)
    np.testing.assert_allclose(got, np.arange(w), atol=1e-5)


def test_depth_png_encoding(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    depth = np.linspace(1.0, 100.0, 64 * 48).reshape(48, 64)
    valid = np.ones_like(depth, dtype=bool)
    valid[:5] = False
    write_depth_png(tmp_path / "d.png", depth, valid)
    back = cv2.imread(str(tmp_path / "d.png"), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16
    assert (back[:5] == 0).all() and (back[5:] > 0).all()
    ratio = back[5:].astype(float) / depth[5:]
    assert np.std(ratio) / np.mean(ratio) < 1e-3  # linear in depth
