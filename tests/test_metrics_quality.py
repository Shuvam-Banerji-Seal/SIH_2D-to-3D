"""Regression tests for point-cloud metrics (F5: chamfer memory blow-up)."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.exceptions import IngestionError
from drone3d.metrics.quality import (
    chamfer_distance,
    cloud_bounds,
    completeness,
    geometric_scale_error,
    point_to_cloud_distances,
    summarize_cloud,
    voxel_coverage,
)


def _brute_force(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Readable O(n*m*3) reference implementation."""
    out = np.empty(len(query))
    for i, q in enumerate(query):
        out[i] = np.sqrt(((reference - q) ** 2).sum(axis=1)).min()
    return out


def test_distances_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    query = rng.normal(size=(17, 3))
    reference = rng.normal(size=(23, 3))

    got = point_to_cloud_distances(query, reference)

    np.testing.assert_allclose(got, _brute_force(query, reference), rtol=1e-12)


def test_identical_points_are_zero() -> None:
    cloud = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    distances = point_to_cloud_distances(cloud, cloud)

    np.testing.assert_allclose(distances, [0.0, 0.0], atol=1e-12)


def test_chunk_clamping_path_still_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memory-budget clamp must not change the result.

    Shrink the budget so ``max_chunk`` falls below the default ``chunk`` and the
    clamped loop -- the replacement for the (chunk, n_ref, 3) broadcast -- is
    exercised on small data.
    """
    monkeypatch.setattr("drone3d.metrics.quality._MEMORY_BUDGET_BYTES", 1024)
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(23, 3))
    query = rng.normal(size=(17, 3))

    got = point_to_cloud_distances(query, reference, chunk=1_000)

    np.testing.assert_allclose(got, _brute_force(query, reference), rtol=1e-12)


def test_empty_clouds_raise() -> None:
    from drone3d.exceptions import IngestionError

    with pytest.raises(IngestionError):
        point_to_cloud_distances(np.empty((0, 3)), np.zeros((3, 3)))
    with pytest.raises(IngestionError):
        point_to_cloud_distances(np.zeros((3, 3)), np.empty((0, 3)))


def test_chamfer_is_symmetric_and_non_negative() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(40, 3))
    b = a + rng.normal(scale=0.1, size=(40, 3))

    ab = chamfer_distance(a, b, max_samples=1_000)
    ba = chamfer_distance(b, a, max_samples=1_000)

    assert ab["chamfer_mean_m"] == pytest.approx(ba["chamfer_mean_m"], rel=1e-9)
    assert ab["chamfer_rmse_m"] == pytest.approx(ba["chamfer_rmse_m"], rel=1e-9)
    assert ab["chamfer_mean_m"] >= 0.0
    assert ab["chamfer_rmse_m"] >= 0.0
    assert set(ab) == {
        "chamfer_mean_m",
        "chamfer_rmse_m",
        "accuracy_mean_m",
        "completeness_mean_m",
    }


# --- criterion 3: bounds, voxel coverage, completeness ----------------------


def test_cloud_bounds() -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])

    bounds = cloud_bounds(points)

    assert bounds["n_points"] == 3
    assert bounds["min"] == [0.0, 0.0, 0.0]
    assert bounds["max"] == [2.0, 4.0, 6.0]
    assert bounds["extent"] == [2.0, 4.0, 6.0]
    assert bounds["max_extent_m"] == 6.0
    assert bounds["centroid"] == pytest.approx([1.0, 5 / 3, 7 / 3])


def test_cloud_bounds_rejects_empty() -> None:
    with pytest.raises(IngestionError):
        cloud_bounds(np.empty((0, 3)))


def test_voxel_coverage_counts_occupied_cells() -> None:
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [10.0, 10.0, 10.0]])

    coverage = voxel_coverage(points, voxel_size=1.0)

    assert coverage["voxel_size_m"] == 1.0
    assert coverage["occupied_voxels"] == 2  # two share a cell, one is alone
    assert coverage["volume_m3"] == 2.0
    assert coverage["points_per_m3"] == 1.5


def test_voxel_coverage_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        voxel_coverage(np.zeros((2, 3)), voxel_size=0.0)
    with pytest.raises(IngestionError):
        voxel_coverage(np.empty((0, 3)), voxel_size=1.0)


def test_completeness_perfect_match() -> None:
    cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    metrics = completeness(cloud, cloud, distance_threshold_m=0.5)

    assert metrics["completeness_ratio"] == 1.0
    assert metrics["mean_distance_m"] == 0.0
    assert metrics["distance_threshold_m"] == 0.5


def test_completeness_partial_match() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    predicted = np.array([[0.0, 0.0, 0.0]])

    metrics = completeness(reference, predicted, distance_threshold_m=0.5)

    # one reference point is recovered, the other is 100 m away
    assert metrics["completeness_ratio"] == 0.5
    assert metrics["mean_distance_m"] == 50.0


def test_geometric_scale_error() -> None:
    metrics = geometric_scale_error(measured_m=102.0, expected_m=100.0)

    assert metrics["absolute_error_m"] == pytest.approx(2.0)
    assert metrics["relative_error"] == pytest.approx(0.02)


def test_geometric_scale_error_rejects_nonpositive_expected() -> None:
    with pytest.raises(ValueError):
        geometric_scale_error(1.0, 0.0)


def test_summarize_cloud_without_reference() -> None:
    summary = summarize_cloud(np.zeros((5, 3)) + np.arange(5)[:, None], voxel_size=1.0)

    assert summary["bounds"]["n_points"] == 5
    assert summary["coverage"]["occupied_voxels"] >= 1
    assert "accuracy_vs_reference" not in summary


def test_summarize_cloud_with_reference_includes_criterion_metrics() -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(30, 3))

    summary = summarize_cloud(points, voxel_size=1.0, reference=points)

    assert summary["completeness_vs_reference"]["completeness_ratio"] == 1.0
    # self-distance is 0 up to the cancellation round-off of the BLAS expansion
    assert summary["accuracy_vs_reference"]["chamfer_mean_m"] == pytest.approx(0.0, abs=1e-6)
