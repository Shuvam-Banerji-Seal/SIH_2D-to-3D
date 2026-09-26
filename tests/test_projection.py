"""Tests for camera projection helpers and the local tangent plane."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.geo.projection import (
    CameraIntrinsics,
    LocalTangentPlane,
    ground_sampling_distance,
    intrinsics_from_fov,
)

# --- intrinsics ------------------------------------------------------------


def test_intrinsics_from_fov_centres_principal_point() -> None:
    intrinsics = intrinsics_from_fov(4000, 3000, hfov_deg=90.0)

    assert intrinsics.cx == pytest.approx((4000 - 1) / 2)
    assert intrinsics.cy == pytest.approx((3000 - 1) / 2)
    assert intrinsics.width == 4000
    assert intrinsics.height == 3000


def test_intrinsics_from_fov_90deg_gives_fx_equals_half_width() -> None:
    intrinsics = intrinsics_from_fov(4000, 3000, hfov_deg=90.0)

    # tan(45 deg) == 1, so fx == width / 2
    assert intrinsics.fx == pytest.approx(2000.0)
    assert intrinsics.fy == pytest.approx(intrinsics.fx)  # square pixels


def test_intrinsics_narrower_fov_gives_longer_focal_length() -> None:
    wide = intrinsics_from_fov(4000, 3000, hfov_deg=120.0)
    narrow = intrinsics_from_fov(4000, 3000, hfov_deg=45.0)

    assert narrow.fx > wide.fx


@pytest.mark.parametrize("bad", [0.0, -10.0, 180.0, 200.0])
def test_intrinsics_rejects_out_of_range_fov(bad: float) -> None:
    with pytest.raises(ValueError):
        intrinsics_from_fov(4000, 3000, hfov_deg=bad)


def test_camera_intrinsics_matrix_and_dict() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=200.0, cx=50.0, cy=25.0, width=10, height=5)

    np.testing.assert_allclose(
        intrinsics.matrix,
        [[100.0, 0.0, 50.0], [0.0, 200.0, 25.0], [0.0, 0.0, 1.0]],
    )
    assert intrinsics.to_dict()["fx"] == 100.0
    assert intrinsics.to_dict()["width"] == 10


# --- ground sampling distance ---------------------------------------------


def test_ground_sampling_distance_scales_with_altitude() -> None:
    low = ground_sampling_distance(100.0)
    high = ground_sampling_distance(200.0)

    assert high == pytest.approx(2 * low)


def test_ground_sampling_distance_known_value() -> None:
    # 100 m AGL, 6.3 mm sensor, 4.5 mm focal, 4000 px -> 0.035 m/px
    assert ground_sampling_distance(100.0) == pytest.approx(0.035, rel=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_ground_sampling_distance_rejects_bad_altitude(bad: float) -> None:
    with pytest.raises(ValueError):
        ground_sampling_distance(bad)


def test_ground_sampling_distance_rejects_bad_sensor() -> None:
    with pytest.raises(ValueError):
        ground_sampling_distance(100.0, sensor_width_mm=0.0)
    with pytest.raises(ValueError):
        ground_sampling_distance(100.0, focal_length_mm=0.0)
    with pytest.raises(ValueError):
        ground_sampling_distance(100.0, image_width_px=0)


# --- local tangent plane ---------------------------------------------------


def test_plane_origin_maps_to_zero() -> None:
    plane = LocalTangentPlane(12.5, 77.6, 500.0)

    np.testing.assert_allclose(plane.to_local(12.5, 77.6, 500.0), [0.0, 0.0, 0.0], atol=1e-6)


def test_plane_round_trip() -> None:
    plane = LocalTangentPlane(12.5, 77.6, 500.0)
    lat, lon, alt = 12.51, 77.62, 650.0

    local = plane.to_local(lat, lon, alt)
    back = plane.to_geodetic(local)

    assert back[0] == pytest.approx(lat, abs=1e-7)
    assert back[1] == pytest.approx(lon, abs=1e-7)
    assert back[2] == pytest.approx(alt, abs=1e-3)


def test_plane_up_matches_altitude_difference() -> None:
    plane = LocalTangentPlane(12.5, 77.6, 500.0)

    local = plane.to_local(12.5, 77.6, 650.0)

    assert local[2] == pytest.approx(150.0, abs=1e-6)


def test_plane_to_dict() -> None:
    plane = LocalTangentPlane(1.0, 2.0, 3.0)

    assert plane.to_dict() == {"lat0": 1.0, "lon0": 2.0, "alt0": 3.0}
