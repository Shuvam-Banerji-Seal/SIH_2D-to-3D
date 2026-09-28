"""Burnt-in overlays: found where a logo sits over a moving scene, not where the scene merely stays in place."""

from __future__ import annotations

import numpy as np

from drone3d.io.overlay import load_mask, save_mask, static_overlay_mask


def _frames(n: int = 12, h: int = 270, w: int = 480, seed: int = 0) -> list[np.ndarray]:
    """A textured scene panning a different amount each frame."""
    rng = np.random.default_rng(seed)
    world = rng.normal(120, 40, (h, 4 * w)).clip(0, 255)
    from scipy import ndimage

    world = ndimage.gaussian_filter(world, 1.5)
    return [world[:, 60 * k : 60 * k + w].copy() for k in range(n)]


def _logo(f: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """A semi-transparent white glyph block (a ring with a bar) in the bottom-left corner."""
    yy, xx = np.mgrid[0:30, 0:30]
    ring = (np.hypot(yy - 15, xx - 15) < 13) & (np.hypot(yy - 15, xx - 15) > 8)
    glyph = ring | ((abs(yy - 15) < 2) & (xx > 4) & (xx < 26))
    out = f.copy()
    region = out[230:260, 12:42]
    region[glyph] = (1 - alpha) * region[glyph] + alpha * 255
    return out


def test_a_semi_transparent_logo_over_a_moving_scene_is_masked() -> None:
    mask = static_overlay_mask([_logo(f) for f in _frames()])
    assert mask is not None
    assert mask[245, 27] and mask.mean() < 0.02  # the logo, and little else
    assert not mask[:200].any()


def test_a_scene_without_overlay_gives_no_mask() -> None:
    assert static_overlay_mask(_frames()) is None


def test_a_level_horizon_that_stays_put_is_not_an_overlay() -> None:
    frames = _frames()
    for f in frames:
        f[:100] = 200.0  # sky: flat, and a horizon edge at the same row in every frame
    assert static_overlay_mask(frames) is None


def test_a_static_shot_is_left_alone() -> None:
    f = _logo(_frames(1)[0])
    assert static_overlay_mask([f] * 12) is None


def test_the_mask_round_trips_through_the_dataset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mask = np.zeros((54, 96), bool)
    mask[40:50, 2:12] = True
    save_mask(tmp_path, mask)
    assert np.array_equal(load_mask(tmp_path), mask)
    assert load_mask(tmp_path, (192, 108))[90, 10]
    save_mask(tmp_path, None)
    assert load_mask(tmp_path) is None
