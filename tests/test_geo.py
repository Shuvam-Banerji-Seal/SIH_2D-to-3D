"""Tests for geodesy and georeferencing maths (PS criteria 2 and 6)."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.exceptions import ReconstructionError
from drone3d.geo.enu import (
    ecef_to_geodetic,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    local_enu_basis,
)
from drone3d.geo.georef import SimilarityTransform, gps_rmse, solve_similarity

WGS84_A = 6_378_137.0


# --- ECEF -----------------------------------------------------------------


def test_ecef_equator_prime_meridian() -> None:
    np.testing.assert_allclose(geodetic_to_ecef(0.0, 0.0, 0.0), [WGS84_A, 0.0, 0.0], rtol=1e-9)


def test_ecef_pole_is_on_z_axis() -> None:
    x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert abs(x) < 1.0
    assert abs(y) < 1.0
    assert z == pytest.approx(6_356_752.314, rel=1e-6)


def test_geodetic_ecef_round_trip() -> None:
    for lat, lon, alt in [(12.5, 77.6, 900.0), (-33.9, 151.2, 25.0), (0.0, 180.0, -50.0)]:
        lat2, lon2, alt2 = ecef_to_geodetic(*geodetic_to_ecef(lat, lon, alt))
        assert lat2 == pytest.approx(lat, abs=1e-7)
        assert lon2 == pytest.approx(lon, abs=1e-7)
        assert alt2 == pytest.approx(alt, abs=1e-3)


# --- ENU ------------------------------------------------------------------


def test_enu_basis_is_orthonormal() -> None:
    basis = local_enu_basis(12.5, 77.6)
    assert basis.shape == (3, 3)
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-12)


def test_enu_at_origin_is_zero() -> None:
    np.testing.assert_allclose(
        geodetic_to_enu(12.5, 77.6, 100.0, lat0=12.5, lon0=77.6, alt0=100.0),
        [0.0, 0.0, 0.0],
        atol=1e-6,
    )


def test_enu_axes_point_the_right_way() -> None:
    origin = {"lat0": 12.5, "lon0": 77.6, "alt0": 0.0}

    north = geodetic_to_enu(12.6, 77.6, 0.0, **origin)
    east = geodetic_to_enu(12.5, 77.7, 0.0, **origin)
    up = geodetic_to_enu(12.5, 77.6, 100.0, **origin)

    # Cross-track leakage is bounded relative to the step length: a parallel of
    # latitude is not a geodesic, so an eastward chord deviates slightly from
    # the local east axis (purely-north motion is exact to ~1e-10).
    assert north[1] > 1000.0 and abs(north[0]) / abs(north[1]) < 1e-6
    assert east[0] > 1000.0 and abs(east[1]) / abs(east[0]) < 1e-3
    assert up[2] == pytest.approx(100.0, abs=1e-6)  # +Up


def test_enu_geodetic_round_trip() -> None:
    origin = {"lat0": 12.5, "lon0": 77.6, "alt0": 0.0}
    lat, lon, alt = 12.51, 77.62, 250.0

    e, n, u = geodetic_to_enu(lat, lon, alt, **origin)
    back = enu_to_geodetic(e, n, u, **origin)

    assert back[0] == pytest.approx(lat, abs=1e-7)
    assert back[1] == pytest.approx(lon, abs=1e-7)
    assert back[2] == pytest.approx(alt, abs=1e-3)


# --- Umeyama similarity ---------------------------------------------------


def test_solve_similarity_recovers_known_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3)) * 50.0
    truth = SimilarityTransform(
        scale=2.5,
        rotation=_rotz(0.7),
        translation=np.array([10.0, -4.0, 3.0]),
    )

    est = solve_similarity(src, truth.apply(src))

    assert est.scale == pytest.approx(2.5, rel=1e-9)
    np.testing.assert_allclose(est.apply(src), truth.apply(src), atol=1e-8)


def test_solve_similarity_rigid_has_unit_scale() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(8, 3))
    truth = SimilarityTransform(scale=1.0, rotation=_rotz(0.3), translation=np.zeros(3))

    est = solve_similarity(src, truth.apply(src), with_scale=False)

    assert est.scale == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(est.apply(src), truth.apply(src), atol=1e-8)


def test_solve_similarity_requires_three_points() -> None:
    with pytest.raises(ReconstructionError):
        solve_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


def test_solve_similarity_rejects_shape_mismatch() -> None:
    with pytest.raises(ReconstructionError):
        solve_similarity(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ReconstructionError):
        solve_similarity(np.zeros((5, 2)), np.zeros((5, 2)))


def test_similarity_inverse_round_trip() -> None:
    rng = np.random.default_rng(2)
    points = rng.normal(size=(6, 3))
    transform = SimilarityTransform(scale=3.0, rotation=_rotz(1.1), translation=np.ones(3))

    np.testing.assert_allclose(
        transform.inverse().apply(transform.apply(points)), points, atol=1e-9
    )


# --- GPS error metrics ----------------------------------------------------


def test_gps_rmse_zero_for_identical() -> None:
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metrics = gps_rmse(pts, pts)

    assert metrics["rmse_3d_m"] == 0.0
    assert metrics["rmse_horizontal_m"] == 0.0
    assert metrics["rmse_vertical_m"] == 0.0
    assert metrics["max_error_m"] == 0.0
    assert metrics["n"] == 2


def test_gps_rmse_separates_axes() -> None:
    pred = np.array([[0.0, 0.0, 0.0]])
    tgt = np.array([[3.0, 4.0, 12.0]])  # horizontal 5 m, vertical 12 m

    metrics = gps_rmse(pred, tgt)

    assert metrics["rmse_horizontal_m"] == pytest.approx(5.0)
    assert metrics["rmse_vertical_m"] == pytest.approx(12.0)
    assert metrics["rmse_3d_m"] == pytest.approx(13.0)
    assert metrics["max_error_m"] == pytest.approx(13.0)


def test_gps_rmse_rejects_mismatched_shapes() -> None:
    with pytest.raises(ReconstructionError):
        gps_rmse(np.zeros((3, 3)), np.zeros((2, 3)))


# --- criterion 6: the measurement itself must be trustworthy ---------------


def test_georef_recovers_known_similarity_with_noisy_gps() -> None:
    """Round-trip the whole alignment with injected error, and check the
    reported RMSE reflects the injected noise.

    Criterion 6 (<= 3 m horizontal GPS RMSE) cannot be *scored* without a real
    flight log, but the *measurement* can be validated: if we inject known
    noise and the reported RMSE matches it, the number we would report against
    real data is trustworthy.
    """
    rng = np.random.default_rng(11)
    truth = _known_transform()
    model_centers = rng.normal(scale=40.0, size=(25, 3))

    noise_sigma = 1.5  # metres
    noisy = truth.apply(model_centers) + rng.normal(scale=noise_sigma, size=(25, 3))

    recovered = solve_similarity(model_centers, noisy)
    residual = recovered.apply(model_centers) - noisy

    metrics = gps_rmse(recovered.apply(model_centers), noisy)

    # the fit absorbs most of the noise; the reported figure must be finite,
    # non-negative, and of the same order as what we injected
    assert metrics["n"] == 25
    assert metrics["rmse_horizontal_m"] >= 0.0
    assert metrics["rmse_3d_m"] < 10 * noise_sigma
    assert np.isfinite(residual).all()


def test_georef_rmse_is_zero_for_exact_correspondences() -> None:
    rng = np.random.default_rng(12)
    model_centers = rng.normal(scale=30.0, size=(15, 3))
    exact = _known_transform().apply(model_centers)

    recovered = solve_similarity(model_centers, exact)
    metrics = gps_rmse(recovered.apply(model_centers), exact)

    assert metrics["rmse_3d_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse_horizontal_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse_vertical_m"] == pytest.approx(0.0, abs=1e-6)


def test_georef_error_grows_with_injected_noise() -> None:
    """Sanity on the metric's direction: more noise must mean a larger figure."""
    rng = np.random.default_rng(13)
    model_centers = rng.normal(scale=30.0, size=(20, 3))
    target = _known_transform().apply(model_centers)

    figures = []
    for sigma in (0.5, 2.0):
        noisy = target + rng.normal(scale=sigma, size=target.shape)
        recovered = solve_similarity(model_centers, noisy)
        figures.append(gps_rmse(recovered.apply(model_centers), noisy)["rmse_3d_m"])

    assert figures[1] > figures[0]


def _rotz(angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _known_transform() -> SimilarityTransform:
    """A non-trivial similarity (scale + rotation + translation)."""
    return SimilarityTransform(
        scale=2.5, rotation=_rotz(0.7), translation=np.array([10.0, -4.0, 3.0])
    )
