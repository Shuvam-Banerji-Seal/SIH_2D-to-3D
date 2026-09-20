"""Per-frame quality measurement and keyframe selection for single-pass video."""

from __future__ import annotations

from bisect import bisect_left, insort

import cv2
import numpy as np

from drone3d.logging_utils import get_logger
from drone3d.types import FrameQuality, FrameRecord

__all__ = [
    "frame_quality",
    "laplacian_variance",
    "measure_frames",
    "score_frame",
    "select_frames",
    "tenengrad",
]

log = get_logger(__name__)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def laplacian_variance(image: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian (higher is sharper)."""
    gray = _as_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(image: np.ndarray) -> float:
    """Focus measure based on the mean squared Sobel gradient magnitude."""
    gray = _as_gray(image).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def frame_quality(
    image: np.ndarray,
    *,
    blur_threshold: float = 20.0,
    max_clipped_fraction: float = 0.5,
) -> FrameQuality:
    """Measure sharpness, exposure and contrast of a BGR frame."""
    gray = _as_gray(image)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(gray.size) or 1.0
    overexposed = float(histogram[250:].sum()) / total
    underexposed = float(histogram[:6].sum()) / total
    sharpness = laplacian_variance(gray)
    usable = (
        sharpness >= blur_threshold
        and overexposed <= max_clipped_fraction
        and underexposed <= max_clipped_fraction
    )
    return FrameQuality(
        sharpness=sharpness,
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        overexposed_ratio=overexposed,
        underexposed_ratio=underexposed,
        usable=usable,
    )


def score_frame(quality: FrameQuality) -> float:
    """Composite 0..1 score used to rank candidate keyframes."""
    sharpness_term = quality.sharpness / (quality.sharpness + 100.0)
    exposure_term = 1.0 - min(1.0, quality.overexposed_ratio + quality.underexposed_ratio)
    contrast_term = min(1.0, quality.contrast / 64.0)
    return float(0.6 * sharpness_term + 0.25 * exposure_term + 0.15 * contrast_term)


def measure_frames(
    records: list[FrameRecord],
    *,
    blur_threshold: float = 20.0,
    max_clipped_fraction: float = 0.5,
) -> list[FrameRecord]:
    """Read each frame from disk and attach a :class:`FrameQuality` measurement."""
    for record in records:
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("cannot read frame, marking unusable: %s", record.path)
            record.quality = FrameQuality(usable=False)
            continue
        record.quality = frame_quality(
            image,
            blur_threshold=blur_threshold,
            max_clipped_fraction=max_clipped_fraction,
        )
    return records


def select_frames(
    records: list[FrameRecord],
    *,
    max_frames: int = 600,
    min_sharpness: float = 0.0,
    min_spacing_s: float = 0.0,
    selection: str = "sharpness",
) -> list[FrameRecord]:
    """Choose a compact, well-spaced subset of frames as SfM input.

    Args:
        records: Frames with optional quality measurements.
        max_frames: Upper bound on the selection size.
        min_sharpness: Reject frames below this Laplacian variance.
        min_spacing_s: Minimum temporal separation between accepted frames.
        selection: ``sharpness`` (greedy, best first) or ``uniform``.

    Returns:
        Selected frames ordered by timestamp.
    """
    if max_frames < 1:
        return []

    candidates = []
    for record in records:
        quality = record.quality
        if quality is not None and (not quality.usable or quality.sharpness < min_sharpness):
            continue
        candidates.append(record)
    if not candidates:
        log.warning("no frames passed quality filtering; falling back to all frames")
        candidates = list(records)
    if len(candidates) <= max_frames and min_spacing_s <= 0:
        return sorted(candidates, key=lambda record: record.timestamp_s)

    if selection == "uniform":
        ranked = candidates
    else:
        ranked = sorted(
            candidates,
            key=lambda record: score_frame(record.quality) if record.quality else 0.5,
            reverse=True,
        )

    accepted: list[FrameRecord] = []
    accepted_times: list[float] = []
    for record in ranked:
        if len(accepted) >= max_frames:
            break
        if min_spacing_s > 0 and accepted_times:
            position = bisect_left(accepted_times, record.timestamp_s)
            left = accepted_times[position - 1] if position > 0 else None
            right = accepted_times[position] if position < len(accepted_times) else None
            if (left is not None and record.timestamp_s - left < min_spacing_s) or (
                right is not None and right - record.timestamp_s < min_spacing_s
            ):
                continue
        insort(accepted_times, record.timestamp_s)
        accepted.append(record)

    accepted.sort(key=lambda record: record.timestamp_s)
    log.info(
        "selected %d/%d frames (selection=%s, min_spacing=%.2fs)",
        len(accepted),
        len(records),
        selection,
        min_spacing_s,
    )
    return accepted
