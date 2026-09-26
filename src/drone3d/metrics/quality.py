"""Metric quality assessment for point clouds, meshes and georeferencing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drone3d.exceptions import IngestionError
from drone3d.logging_utils import get_logger
from drone3d.utils.ply import load_ply

__all__ = [
    "chamfer_distance",
    "cloud_bounds",
    "completeness",
    "geometric_scale_error",
    "point_to_cloud_distances",
    "summarize_cloud",
    "voxel_coverage",
]

log = get_logger(__name__)

_MAX_SAMPLES = 50_000
_CHUNK = 1_000
# Upper bound on the ``(chunk, n_ref)`` scratch block used by
# `point_to_cloud_distances`, so reference size cannot blow up memory.
_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024


def _as_points(points: Any) -> np.ndarray:
    if isinstance(points, str | Path):
        return load_ply(points).points
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _subsample(points: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or len(points) <= max_samples:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_samples).round().astype(int)
    return points[np.unique(indices)]


def cloud_bounds(points: Any) -> dict[str, Any]:
    """Axis-aligned bounds, extent and centroid of a cloud."""
    array = _as_points(points)
    if len(array) == 0:
        raise IngestionError("cannot compute bounds of an empty point cloud")
    minimum = array.min(axis=0)
    maximum = array.max(axis=0)
    extent = maximum - minimum
    return {
        "n_points": int(len(array)),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "extent": extent.tolist(),
        "max_extent_m": float(extent.max()),
        "centroid": array.mean(axis=0).tolist(),
    }


def voxel_coverage(points: Any, voxel_size: float) -> dict[str, Any]:
    """Occupied-voxel count, volume and density of a cloud."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    array = _as_points(points)
    if len(array) == 0:
        raise IngestionError("cannot compute voxel coverage of an empty point cloud")
    keys = np.floor(array / voxel_size).astype(np.int64)
    unique = np.unique(keys, axis=0)
    volume = float(len(unique)) * voxel_size**3
    return {
        "voxel_size_m": float(voxel_size),
        "occupied_voxels": int(len(unique)),
        "volume_m3": round(volume, 4),
        "points_per_m3": round(len(array) / volume, 3) if volume > 0 else 0.0,
    }


def point_to_cloud_distances(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk: int = _CHUNK,
) -> np.ndarray:
    """Distance from every query point to the nearest reference point (chunked).

    Uses ``|a-b|^2 = |a|^2 + |b|^2 - 2 a·b`` so a chunk never materialises a
    ``(chunk, n_ref, 3)`` broadcast array (which, with ``deltas**2``, cost two
    such arrays at once -- ~2.4 GB at the default 50k reference points). The
    chunk size is clamped so the ``(chunk, n_ref)`` scratch block stays inside
    :data:`_MEMORY_BUDGET_BYTES`.
    """
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(reference) == 0 or len(query) == 0:
        raise IngestionError("both clouds must be non-empty")

    ref_sq = np.einsum("ij,ij->i", reference, reference)
    max_chunk = max(1, _MEMORY_BUDGET_BYTES // (len(reference) * np.dtype(np.float64).itemsize))
    step = max(1, min(chunk, max_chunk, len(query)))

    distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), step):
        block = query[start : start + step]
        block_sq = np.einsum("ij,ij->i", block, block)
        sq = block_sq[:, None] + ref_sq[None, :] - 2.0 * (block @ reference.T)
        # Cancellation can make an exact zero slightly negative.
        np.maximum(sq, 0.0, out=sq)
        distances[start : start + step] = np.sqrt(sq.min(axis=1))
    return distances


def chamfer_distance(
    predicted: Any,
    reference: Any,
    *,
    max_samples: int = _MAX_SAMPLES,
) -> dict[str, float]:
    """Symmetric Chamfer distance between two clouds (metres)."""
    pred = _subsample(_as_points(predicted), max_samples)
    ref = _subsample(_as_points(reference), max_samples)
    forward = point_to_cloud_distances(pred, ref)
    backward = point_to_cloud_distances(ref, pred)
    return {
        "chamfer_mean_m": float(0.5 * (forward.mean() + backward.mean())),
        "chamfer_rmse_m": float(np.sqrt(0.5 * (np.mean(forward**2) + np.mean(backward**2)))),
        "accuracy_mean_m": float(forward.mean()),
        "completeness_mean_m": float(backward.mean()),
    }


def completeness(
    reference: Any,
    predicted: Any,
    *,
    distance_threshold_m: float = 0.5,
    max_samples: int = _MAX_SAMPLES,
) -> dict[str, float]:
    """Fraction of reference points reconstructed within ``distance_threshold_m``."""
    ref = _subsample(_as_points(reference), max_samples)
    pred = _subsample(_as_points(predicted), max_samples)
    distances = point_to_cloud_distances(ref, pred)
    return {
        "distance_threshold_m": float(distance_threshold_m),
        "completeness_ratio": float(np.mean(distances <= distance_threshold_m)),
        "mean_distance_m": float(distances.mean()),
    }


def geometric_scale_error(measured_m: float, expected_m: float) -> dict[str, float]:
    """Absolute and relative error between a measured and expected distance."""
    if expected_m <= 0:
        raise ValueError("expected_m must be positive")
    error = measured_m - expected_m
    return {
        "measured_m": float(measured_m),
        "expected_m": float(expected_m),
        "absolute_error_m": float(error),
        "relative_error": float(error / expected_m),
    }


def summarize_cloud(
    points: Any,
    *,
    voxel_size: float = 0.5,
    reference: Any | None = None,
) -> dict[str, Any]:
    """Convenience summary used by the metrics pipeline stage."""
    array = _subsample(_as_points(points), _MAX_SAMPLES)
    summary: dict[str, Any] = {"bounds": cloud_bounds(array)}
    summary["coverage"] = voxel_coverage(array, voxel_size)
    if reference is not None:
        summary["accuracy_vs_reference"] = chamfer_distance(array, reference)
        summary["completeness_vs_reference"] = completeness(reference, array)
    return summary
