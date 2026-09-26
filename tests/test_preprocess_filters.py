"""Tests for stabilization, Wiener deblur and dynamic-object masking."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drone3d.preprocess.deblur import auto_deblur, wiener_deconvolution
from drone3d.preprocess.dynamic import DynamicMasker, denoise_mask
from drone3d.preprocess.quality import laplacian_variance
from drone3d.preprocess.stabilize import (
    Stabilizer,
    _compose,
    estimate_affine,
    smooth_trajectory,
    stabilize_frames,
)


def _texture(size: int = 128, seed: int = 0) -> np.ndarray:
    """A textured BGR frame with plenty of ORB-detectable corners.

    Returned as 3-channel BGR because that is what the stabilizer expects.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    board = (((xx // 12) + (yy // 12)) % 2 * 200 + 40).astype(np.uint8)
    single = (
        (board + rng.integers(0, 20, board.shape, dtype=np.uint8)).clip(0, 255).astype(np.uint8)
    )
    return cv2.cvtColor(single, cv2.COLOR_GRAY2BGR)


# --- stabilization ---------------------------------------------------------


def test_compose_multiply_and_drop_last_row() -> None:
    outer = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    inner = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]])

    combined = _compose(outer, inner)

    assert combined.shape == (2, 3)
    # 2 * (x+3) + 1 = 2x + 7 ; 2 * (y+4) + 1 = 2y + 9
    np.testing.assert_allclose(combined, [[2.0, 0.0, 7.0], [0.0, 2.0, 9.0]])


def test_estimate_affine_recovers_pure_translation() -> None:
    base = _texture()
    shifted = np.roll(base, 5, axis=1).copy()

    matrix = estimate_affine(base, shifted)

    assert matrix is not None
    # mapping shifted -> base must undo the roll
    assert abs(matrix[0, 2]) >= 3.0


def test_estimate_affine_returns_none_without_features() -> None:
    flat = np.full((64, 64), 128, dtype=np.uint8)

    assert estimate_affine(flat, flat) is None


def test_smooth_trajectory_empty() -> None:
    assert smooth_trajectory([]) == []


def test_smooth_trajectory_reduces_jitter() -> None:
    rng = np.random.default_rng(3)
    transforms = [
        np.array([[1.0, 0.0, float(i) + rng.normal(0, 2)], [0.0, 1.0, 0.0]]) for i in range(40)
    ]

    smoothed = smooth_trajectory(transforms, window=7)

    assert len(smoothed) == len(transforms)
    raw_dx = np.diff([t[0, 2] for t in transforms])
    smooth_dx = np.diff([t[0, 2] for t in smoothed])
    assert np.std(smooth_dx) < np.std(raw_dx)


def test_stabilize_frames_on_identical_frames_is_identity() -> None:
    frames = [_texture(seed=1)] * 5

    out = stabilize_frames(frames)

    assert len(out) == len(frames)
    np.testing.assert_allclose(out[0], frames[0])


def test_stabilizer_class_runs_per_frame() -> None:
    """``Stabilizer.apply`` is a streaming single-frame API, not a batch one."""
    stabilizer = Stabilizer()
    base = _texture(seed=2)
    frames = [base, np.roll(base, 3, axis=1).copy(), np.roll(base, 6, axis=1).copy()]

    out = [stabilizer.apply(frame) for frame in frames]

    assert len(out) == len(frames)
    for frame in out:
        assert frame.shape == base.shape


# --- Wiener deblur ---------------------------------------------------------


def test_wiener_deconvolution_shape_and_dtype() -> None:
    image = _texture()

    restored = wiener_deconvolution(image, psf_sigma=1.0, snr=0.01)

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8


def test_wiener_deconvolution_rejects_bad_snr() -> None:
    image = _texture()

    with pytest.raises(ValueError):
        wiener_deconvolution(image, snr=0.0)


def test_auto_deblur_leaves_sharp_images_alone() -> None:
    sharp = _texture()
    assert laplacian_variance(sharp) > 60.0

    assert auto_deblur(sharp, threshold=60.0) is sharp


def test_auto_deblur_never_regresses_sharpness() -> None:
    """The guard: a deblur attempt must not make the frame worse."""
    rng = np.random.default_rng(7)
    soft = rng.integers(90, 110, size=(64, 64), dtype=np.uint8)
    soft = cv2.GaussianBlur(soft, (0, 0), 2.0)
    image = cv2.cvtColor(soft, cv2.COLOR_GRAY2BGR)

    out = auto_deblur(image, threshold=1_000_000.0)  # force the deblur branch

    assert laplacian_variance(out) >= laplacian_variance(image)


# --- dynamic masking -------------------------------------------------------


def test_dynamic_masker_flags_a_moving_object() -> None:
    masker = DynamicMasker()
    background = _texture(seed=11)

    # warm up the background model, then introduce a moving block
    for _ in range(4):
        masker.mask(background)
    moved = background.copy()
    moved[40:80, 40:80] = 255

    mask = masker.mask(moved)

    # the mask is single-channel: 255 marks dynamic pixels
    assert mask.shape == background.shape[:2]
    assert mask.dtype == np.uint8
    assert mask[60, 60] == 255  # the moved region is marked dynamic
    assert set(np.unique(mask)) <= {0, 255}


def test_dynamic_masker_ratio_is_fractional() -> None:
    masker = DynamicMasker()
    frame = _texture(seed=12)

    ratio = masker.moving_ratio(frame)

    assert 0.0 <= ratio <= 1.0


def test_denoise_mask_preserves_large_regions() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255

    cleaned = denoise_mask(mask, min_area=50.0)

    assert cleaned[50, 50] == 255
    assert cleaned.sum() > 0
