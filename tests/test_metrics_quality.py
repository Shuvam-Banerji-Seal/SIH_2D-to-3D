"""Regression tests for point-cloud metrics (F5: chamfer memory blow-up)."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.metrics.quality import chamfer_distance, point_to_cloud_distances


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
