"""Similarity-transform georeferencing of local models to global coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone3d.exceptions import ReconstructionError

__all__ = ["SimilarityTransform", "gps_rmse", "solve_similarity"]


@dataclass
class SimilarityTransform:
    """Rigid transform with uniform scale: ``dst = s * R @ src + t``."""

    scale: float = 1.0
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rotation is None:
            self.rotation = np.eye(3)
        if self.translation is None:
            self.translation = np.zeros(3)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the transform to an ``(N, 3)`` (or ``(3,)``) array of points."""
        arr = np.asarray(points, dtype=np.float64)
        single = arr.ndim == 1
        arr = np.atleast_2d(arr)
        if arr.shape[1] != 3:
            raise ReconstructionError(f"expected (N, 3) points, got {arr.shape}")
        out = self.scale * (arr @ self.rotation.T) + self.translation
        return out[0] if single else out

    def inverse(self) -> SimilarityTransform:
        """Return the inverse transform."""
        inv_scale = 1.0 / self.scale
        inv_rotation = self.rotation.T
        inv_translation = -inv_scale * (inv_rotation @ self.translation)
        return SimilarityTransform(inv_scale, inv_rotation, inv_translation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix": self.to_matrix().tolist(),
        }

    def to_matrix(self) -> np.ndarray:
        """Return the homogeneous 4x4 representation."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def solve_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    with_scale: bool = True,
) -> SimilarityTransform:
    """Least-squares similarity transform mapping ``src`` onto ``dst`` (Umeyama).

    Args:
        src: ``(N, 3)`` source points.
        dst: ``(N, 3)`` target points, same ordering.
        with_scale: Estimate uniform scale (disabled -> rigid transform).

    Raises:
        ReconstructionError: If fewer than 3 correspondences are supplied or
            the inputs disagree in shape.
    """
    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    if src_arr.ndim != 2 or src_arr.shape[1] != 3:
        raise ReconstructionError("src must be an (N, 3) array")
    if dst_arr.shape != src_arr.shape:
        raise ReconstructionError(
            f"src and dst must have identical shapes, got {src_arr.shape} and {dst_arr.shape}"
        )
    if src_arr.shape[0] < 3:
        raise ReconstructionError(
            f"at least 3 correspondences are required, got {src_arr.shape[0]}"
        )

    mean_src = src_arr.mean(axis=0)
    mean_dst = dst_arr.mean(axis=0)
    x_centered = src_arr - mean_src
    y_centered = dst_arr - mean_dst

    covariance = (y_centered.T @ x_centered) / src_arr.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    variance_src = float(np.mean(np.sum(x_centered**2, axis=1)))
    if with_scale and variance_src > 1e-12:
        scale = float(np.trace(np.diag(singular_values) @ correction) / variance_src)
    else:
        scale = 1.0
    translation = mean_dst - scale * (rotation @ mean_src)
    return SimilarityTransform(scale, rotation, translation)


def gps_rmse(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return RMSE and per-axis MAE for georeferenced point correspondences."""
    pred = np.asarray(predicted, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape or pred.ndim != 2:
        raise ReconstructionError("predicted and target must be matching (N, 3) arrays")
    delta = pred - tgt
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    return {
        "rmse_3d_m": float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
        "rmse_horizontal_m": float(np.sqrt(np.mean(horizontal**2))),
        "rmse_vertical_m": float(np.sqrt(np.mean(delta[:, 2] ** 2))),
        "max_error_m": float(np.max(np.linalg.norm(delta, axis=1))),
        "n": int(pred.shape[0]),
    }
