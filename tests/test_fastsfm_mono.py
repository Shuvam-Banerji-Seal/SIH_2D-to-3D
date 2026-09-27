"""Monocular fill: calibrated to triangulated depth, fills the gaps, never the sky."""

from __future__ import annotations

import numpy as np

from drone3d.fastsfm.mono import calibrate_fill


def test_fill_follows_the_calibration_and_skips_sky() -> None:
    h, w = 60, 80
    z_true = np.tile(np.linspace(10.0, 40.0, h)[:, None], (1, w))  # depth grows with the row
    disparity = 100.0 / z_true  # what the prior sees, up to scale
    disparity[:10] = 0.0  # sky: zero disparity
    tri = np.where(np.arange(w)[None, :] % 3 == 0, z_true, 0.0).astype(np.float32)  # every 3rd column triangulated
    tri[:10] = 0.0
    tri[2, 5] = 55.0  # a stray triangulated speck in the sky
    depth, info = calibrate_fill(disparity, tri, min_samples=50)
    assert info["status"] == "filled"
    assert not depth[:10].any()  # sky stays empty, the speck is dropped
    below = depth[10:]
    assert (below > 0).all()
    np.testing.assert_allclose(below, z_true[10:], rtol=0.03)


def test_too_few_samples_leaves_the_map_alone() -> None:
    disparity = np.ones((20, 20))
    tri = np.zeros((20, 20), np.float32)
    tri[0, :5] = 10.0
    depth, info = calibrate_fill(disparity, tri, min_samples=50)
    assert info["status"] == "too-few-samples"
    np.testing.assert_array_equal(depth, tri)
