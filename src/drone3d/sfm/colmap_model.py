"""Parsers for COLMAP text models (camera poses and sparse points)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drone3d.exceptions import ReconstructionError

__all__ = ["ImagePose", "parse_images_text", "parse_points3d", "qvec_to_rotation"]


@dataclass
class ImagePose:
    """A registered image pose from COLMAP's ``images.txt``.

    COLMAP stores the world-to-camera transform: ``x_cam = R(q) @ x_world + t``.
    """

    image_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int

    @property
    def rotation(self) -> np.ndarray:
        return qvec_to_rotation(self.qvec)

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates."""
        return -self.rotation.T @ self.tvec


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP quaternion ``(w, x, y, z)`` to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in np.asarray(qvec, dtype=np.float64).reshape(4))
    norm = w * w + x * x + y * y + z * z
    if norm < 1e-12:
        raise ReconstructionError("degenerate quaternion in COLMAP model")
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z) / norm,
                2 * (x * y - w * z) / norm,
                2 * (x * z + w * y) / norm,
            ],
            [
                2 * (x * y + w * z) / norm,
                1 - 2 * (x * x + z * z) / norm,
                2 * (y * z - w * x) / norm,
            ],
            [
                2 * (x * z - w * y) / norm,
                2 * (y * z + w * x) / norm,
                1 - 2 * (x * x + y * y) / norm,
            ],
        ],
        dtype=np.float64,
    )


def parse_images_text(path: str | Path) -> list[ImagePose]:
    """Parse ``images.txt`` from a COLMAP TXT model; pose lines only."""
    model_path = Path(path)
    if not model_path.is_file():
        raise ReconstructionError(f"COLMAP images.txt not found: {model_path}")

    poses: list[ImagePose] = []
    for line in model_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 10:
            continue  # points2D observation line
        try:
            image_id = int(fields[0])
            qvec = np.array([float(v) for v in fields[1:5]], dtype=np.float64)
            tvec = np.array([float(v) for v in fields[5:8]], dtype=np.float64)
            camera_id = int(fields[8])
        except ValueError:
            continue
        name = " ".join(fields[9:])
        poses.append(ImagePose(image_id, name, qvec, tvec, camera_id))
    return poses


def parse_points3d(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, rgb)`` arrays from a COLMAP ``points3D.txt`` file."""
    model_path = Path(path)
    if not model_path.is_file():
        raise ReconstructionError(f"COLMAP points3D.txt not found: {model_path}")

    xyz: list[list[float]] = []
    rgb: list[list[int]] = []
    for line in model_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        try:
            xyz.append([float(fields[1]), float(fields[2]), float(fields[3])])
            rgb.append([int(fields[4]), int(fields[5]), int(fields[6])])
        except ValueError:
            continue
    return np.array(xyz, dtype=np.float64), np.array(rgb, dtype=np.uint8)
