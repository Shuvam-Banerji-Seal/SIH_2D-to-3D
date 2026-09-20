"""Camera geometry, ground sampling distance and local tangent-plane helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drone3d.geo.enu import enu_to_geodetic, geodetic_to_enu

__all__ = [
    "CameraIntrinsics",
    "LocalTangentPlane",
    "ground_sampling_distance",
    "intrinsics_from_fov",
]


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> CameraIntrinsics:
    """Build intrinsics assuming square pixels from image size and horizontal FoV."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must be in (0, 180)")
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return CameraIntrinsics(
        fx=fx, fy=fx, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, width=width, height=height
    )


def ground_sampling_distance(
    altitude_agl_m: float,
    *,
    sensor_width_mm: float = 6.3,
    focal_length_mm: float = 4.5,
    image_width_px: int = 4000,
) -> float:
    """Ground sampling distance in metres/pixel for a nadir camera."""
    if altitude_agl_m <= 0:
        raise ValueError("altitude_agl_m must be positive")
    if sensor_width_mm <= 0 or focal_length_mm <= 0 or image_width_px <= 0:
        raise ValueError("sensor, focal length and image width must be positive")
    return (altitude_agl_m * sensor_width_mm) / (focal_length_mm * image_width_px)


class LocalTangentPlane:
    """Bidirectional converter between geodetic coordinates and a local ENU frame."""

    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0) -> None:
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.alt0 = float(alt0)

    def to_local(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        """Geodetic degrees/metres -> local East-North-Up metres."""
        return geodetic_to_enu(lat, lon, alt, lat0=self.lat0, lon0=self.lon0, alt0=self.alt0)

    def to_geodetic(self, point: np.ndarray) -> tuple[float, float, float]:
        """Local ENU metres -> ``(lat_deg, lon_deg, alt_m)``."""
        east, north, up = (float(v) for v in np.asarray(point, dtype=np.float64).reshape(3))
        return enu_to_geodetic(east, north, up, lat0=self.lat0, lon0=self.lon0, alt0=self.alt0)

    def to_dict(self) -> dict[str, float]:
        return {"lat0": self.lat0, "lon0": self.lon0, "alt0": self.alt0}
