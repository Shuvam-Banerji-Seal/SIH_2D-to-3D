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


def test_a_wall_where_the_prior_sees_depth_is_rejected_but_flat_ground_is_not() -> None:
    h, w = 60, 80
    z_true = np.tile(np.linspace(10.0, 40.0, h)[:, None], (1, w))
    disparity = 100.0 / z_true  # the prior sees a 4x range of depth
    wall = np.where(np.arange(w)[None, :] % 3 == 0, 20.0 + 0.1 * np.random.default_rng(0).random((h, w)), 0.0).astype(np.float32)
    depth, info = calibrate_fill(disparity, wall, min_samples=50)  # triangulation: one depth everywhere (collapsed poses)
    assert info["status"] == "depth-contradicts-prior" and not depth.any()
    flat = np.full((h, w), 30.0) * (1 + 0.004 * np.random.default_rng(1).random((h, w)))  # nadir over flat ground
    tri = np.where(np.arange(w)[None, :] % 3 == 0, flat, 0.0).astype(np.float32)
    depth, info = calibrate_fill(100.0 / flat, tri, min_samples=50)  # the prior is flat too
    assert info["status"] != "depth-contradicts-prior" and (depth > 0).mean() > 0.9


def test_moge_disparity_keeps_its_own_sky_and_a_far_horizon() -> None:
    """MoGe's masked pixels (NaN) are sky; a horizon 500x farther than the nearest roof stays valid depth."""
    import torch

    from drone3d.fastsfm.mono import MoGeDepth

    class FakeMoGe(torch.nn.Module):
        def infer(self, x, resolution_level=0, use_fp16=True, fov_x=None):  # type: ignore[no-untyped-def]
            b, _, h, w = x.shape
            d = torch.linspace(1.0, 500.0, w).expand(b, h, w).clone()
            d[:, : h // 4] = float("nan")  # the top quarter: sky
            self.fov = fov_x
            return {"depth": d}

    net = FakeMoGe()
    mono = MoGeDepth(net=net, device="cpu")
    frames = torch.zeros(2, 40, 60, 3, dtype=torch.uint8)
    disp = mono(frames, fov_x=torch.tensor([60.0, 61.0])).numpy()
    assert (disp[:, :10] == 0).all() and (disp[:, 10:] > 0).all()
    assert torch.allclose(net.fov, torch.tensor([60.0, 61.0]))
    tri = np.zeros((40, 60), np.float32)
    tri[20:, :30] = np.linspace(1.0, 500.0, 60)[None, :30] * 2.0  # triangulated: twice MoGe's scale, near half
    depth, info = calibrate_fill(disp[0], tri, sky_rel=mono.sky_rel, far_factor=10.0)
    assert info["status"] == "filled"
    assert (depth[:10] == 0).all()  # sky stays empty
    assert depth[30, 55] > 0 and abs(depth[30, 55] / (2.0 * np.linspace(1.0, 500.0, 60)[55]) - 1) < 0.05
