"""Tests for preprocessing: quality scoring, keyframe selection, masks, deblur."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone3d.preprocess.deblur import gaussian_psf, unsharp_mask
from drone3d.preprocess.dynamic import denoise_mask
from drone3d.preprocess.quality import (
    frame_quality,
    laplacian_variance,
    select_frames,
    tenengrad,
)
from drone3d.types import FrameQuality, FrameRecord

# --- quality metrics -------------------------------------------------------


def _checkerboard(size: int = 64, cell: int = 8) -> np.ndarray:
    """A high-frequency pattern: sharp by construction."""
    yy, xx = np.mgrid[0:size, 0:size]
    board = (((xx // cell) + (yy // cell)) % 2 * 255).astype(np.uint8)
    return np.dstack([board] * 3)


def _flat(size: int = 64, value: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_sharp_image_has_higher_laplacian_variance_than_blurred() -> None:
    sharp = laplacian_variance(_checkerboard())
    blurred = laplacian_variance(_flat())

    assert sharp > 10 * max(blurred, 1e-6)


def test_tenengrad_prefers_edges() -> None:
    assert tenengrad(_checkerboard()) > tenengrad(_flat())


def test_frame_quality_flags_flat_image_as_unusable() -> None:
    quality = frame_quality(_flat(), blur_threshold=20.0)

    assert quality.usable is False
    assert quality.sharpness == pytest.approx(0.0)


def test_frame_quality_accepts_sharp_image() -> None:
    quality = frame_quality(_checkerboard(), blur_threshold=20.0)

    assert quality.usable is True
    assert quality.sharpness > 20.0


def test_frame_quality_detects_overexposure() -> None:
    blown = np.full((32, 32, 3), 255, dtype=np.uint8)

    quality = frame_quality(blown, blur_threshold=0.0, max_clipped_fraction=0.5)

    assert quality.overexposed_ratio > 0.9
    assert quality.usable is False


# --- keyframe selection ----------------------------------------------------


def _records(n: int, sharpness: float = 100.0) -> list[FrameRecord]:
    out = []
    for i in range(n):
        record = FrameRecord(index=i, timestamp_s=float(i), path=Path(f"f{i}.jpg"))
        record.quality = FrameQuality(sharpness=sharpness, brightness=128.0, contrast=32.0)
        out.append(record)
    return out


def test_select_frames_respects_max_frames() -> None:
    selected = select_frames(_records(20), max_frames=5)

    assert len(selected) == 5


def test_select_frames_zero_budget_returns_empty() -> None:
    assert select_frames(_records(5), max_frames=0) == []


def test_select_frames_orders_by_timestamp() -> None:
    selected = select_frames(_records(10), max_frames=3)

    times = [r.timestamp_s for r in selected]
    assert times == sorted(times)


def test_select_frames_enforces_min_spacing() -> None:
    selected = select_frames(_records(20), max_frames=20, min_spacing_s=3.0)

    times = [r.timestamp_s for r in selected]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert len(times) >= 2
    assert all(gap >= 3.0 for gap in gaps)


def test_select_frames_prefers_sharper_frames() -> None:
    records = _records(5, sharpness=5.0)
    records[2].quality = FrameQuality(sharpness=500.0, brightness=128.0, contrast=32.0)

    selected = select_frames(records, max_frames=1)

    assert selected[0].index == 2


def test_select_frames_rejects_below_min_sharpness() -> None:
    records = _records(6, sharpness=50.0)
    for record in records[:3]:
        record.quality = FrameQuality(sharpness=1.0, brightness=128.0, contrast=32.0)

    selected = select_frames(records, max_frames=10, min_sharpness=10.0)

    assert [r.index for r in selected] == [3, 4, 5]


def test_select_frames_skips_unusable_but_falls_back_if_all_filtered() -> None:
    records = _records(3)
    for record in records:
        record.quality = FrameQuality(sharpness=999.0, usable=False)

    selected = select_frames(records, max_frames=3, min_sharpness=10.0)

    # every candidate was filtered, so the documented fallback returns them all
    assert len(selected) == 3


def test_select_frames_uniform_mode_is_time_even() -> None:
    selected = select_frames(_records(10), max_frames=5, selection="uniform")

    assert len(selected) == 5


# --- masks -----------------------------------------------------------------


def test_denoise_mask_drops_small_specks() -> None:
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[10:12, 10:12] = 255  # 4 px speck -> below min_area
    mask[50:110, 50:110] = 255  # 3600 px blob -> kept

    cleaned = denoise_mask(mask, min_area=400.0)

    assert cleaned[11, 11] == 0
    assert cleaned[80, 80] == 255


def test_denoise_mask_keeps_empty_mask_empty() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)

    assert denoise_mask(mask).sum() == 0


# --- deblur ----------------------------------------------------------------


def test_gaussian_psf_is_normalized() -> None:
    psf = gaussian_psf((31, 31), sigma=2.0)

    assert psf.sum() == pytest.approx(1.0, rel=1e-6)
    assert psf.min() >= 0.0
    # ifftshift moves the centred peak to the origin, ready for FFT convolution
    assert psf[0, 0] == psf.max()
    assert np.count_nonzero(psf) > 1


def test_unsharp_mask_increases_local_contrast() -> None:
    # Real blur preserves gradient structure; a naive quantising "soften" would
    # collapse the two tone levels and make the test vacuous.
    import cv2

    image = cv2.GaussianBlur(_checkerboard(cell=4), (0, 0), sigmaX=1.2)

    sharpened = unsharp_mask(image, sigma=1.5, amount=1.5)

    assert sharpened.shape == image.shape
    assert laplacian_variance(sharpened) > laplacian_variance(image)
