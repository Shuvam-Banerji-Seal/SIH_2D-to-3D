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
    "load_cloud",
    "point_to_cloud_distances",
    "summarize_cloud",
    "voxel_coverage",
]

log = get_logger(__name__)

_MAX_SAMPLES = (
    500_000  # per cloud, for distances; a KD-tree query of this many takes about a second
)


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
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    if (
        float(np.prod(span.astype(np.float64))) < 2.0**62
    ):  # one int64 per voxel: much faster than unique rows
        occupied = len(np.unique((keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]))
    else:
        occupied = len(np.unique(keys, axis=0))
    volume = float(occupied) * voxel_size**3
    return {
        "voxel_size_m": float(voxel_size),
        "occupied_voxels": int(occupied),
        "volume_m3": round(volume, 4),
        "points_per_m3": round(len(array) / volume, 3) if volume > 0 else 0.0,
    }


def point_to_cloud_distances(
    query: np.ndarray, reference: np.ndarray, *, workers: int = -1
) -> np.ndarray:
    """Exact distance from every query point to the nearest reference point (KD-tree)."""
    from scipy.spatial import cKDTree

    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(reference) == 0 or len(query) == 0:
        raise IngestionError("both clouds must be non-empty")
    distances, _ = cKDTree(reference).query(query, k=1, workers=workers)
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
    distance_threshold_m: float = 0.5,
) -> dict[str, Any]:
    """Bounds and voxel coverage of the whole cloud; distances to ``reference`` on subsamples."""
    array = _as_points(points)
    summary: dict[str, Any] = {"bounds": cloud_bounds(array)}
    summary["coverage"] = voxel_coverage(array, voxel_size)
    if reference is not None:
        summary["accuracy_vs_reference"] = chamfer_distance(array, reference)
        summary["completeness_vs_reference"] = completeness(
            reference, array, distance_threshold_m=distance_threshold_m
        )
    return summary


def load_cloud(path: str | Path, *, target_epsg: int | None = None) -> np.ndarray:
    """A cloud file as ``[N, 3]`` float64.

    LAS/LAZ keep their projected coordinates, reprojected to ``target_epsg``
    when the file carries a CRS that differs; anything else (PLY, PCD, XYZ,
    PTS) is read by Open3D as is.
    """
    path = Path(path)
    if not path.is_file():
        raise IngestionError(f"reference cloud not found: {path}")
    if path.suffix.lower() in (".las", ".laz"):
        import laspy

        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        crs = las.header.parse_crs()
        if crs is not None and target_epsg is not None and crs.to_epsg() != target_epsg:
            import pyproj

            to = pyproj.Transformer.from_crs(crs, f"EPSG:{target_epsg}", always_xy=True)
            pts = np.column_stack(to.transform(pts[:, 0], pts[:, 1], pts[:, 2]))
        return pts
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)


def scene_view_completeness(sfm: dict, dense: dict) -> float | None:
    """The share of all registered keyframes' non-sky pixels the meshes cover.

    Each SfM model counts with its registered images, whether or not the dense
    stage made a mesh of it: a model without one (too few views, no depth)
    contributes zeros, not nothing. Its own view completeness is measured on the
    keyframes dense used (every second one for large models), which stand for
    all of the model's views. ``sfm``/``dense`` are the stages' ``result.json``.
    """
    reg = {str(m["path"]).rstrip("/").rsplit("/", 1)[-1]: m["images"] for m in sfm.get("models", [])}
    total = sum(reg.values())
    if not total:
        return None
    got = {str(m["model"]).rstrip("/").rsplit("/", 1)[-1]: m["view_completeness"] for m in dense.get("models", [])
           if m.get("status") == "ok" and m.get("view_completeness") is not None}  # fmt: skip
    return round(sum(reg[k] * c for k, c in got.items() if k in reg) / total, 4)
