"""WGS84 geodetic <-> ECEF <-> local ENU conversions (pure NumPy)."""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ecef_to_geodetic",
    "enu_to_geodetic",
    "geodetic_to_ecef",
    "geodetic_to_enu",
    "local_enu_basis",
]

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Convert geodetic coordinates to Earth-Centered Earth-Fixed metres."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(
    x: float, y: float, z: float, iterations: int = 8
) -> tuple[float, float, float]:
    """Convert ECEF metres to ``(lat_deg, lon_deg, alt_m)`` using Bowring iteration."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z)
        return math.degrees(lat), math.degrees(lon), abs(z) - WGS84_B

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(iterations):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def local_enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the 3x3 rotation whose rows are local East, North, Up axes (ECEF)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.vstack([east, north, up])


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0,
    *,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> np.ndarray:
    """Convert a geodetic point to local East-North-Up metres about ``(lat0, lon0, alt0)``."""
    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    ecef0 = geodetic_to_ecef(lat0, lon0, alt0)
    return local_enu_basis(lat0, lon0) @ (ecef - ecef0)


def enu_to_geodetic(
    east_m: float,
    north_m: float,
    up_m: float,
    *,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> tuple[float, float, float]:
    """Inverse of :func:`geodetic_to_enu`."""
    delta = local_enu_basis(lat0, lon0).T @ np.array([east_m, north_m, up_m], dtype=np.float64)
    ecef0 = geodetic_to_ecef(lat0, lon0, alt0)
    x, y, z = ecef0 + delta
    return ecef_to_geodetic(float(x), float(y), float(z))
