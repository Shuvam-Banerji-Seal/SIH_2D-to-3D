"""Similarity-transform georeferencing of local models to global coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone3d.exceptions import ReconstructionError

__all__ = [
    "SimilarityTransform",
    "estimate_up",
    "gps_rmse",
    "leave_one_out",
    "solve_georef",
    "solve_similarity",
    "solve_yaw_scale",
    "track_shape",
]


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


# --------------------------------------------------------------------------
# Single-pass tracks are nearly straight lines. Camera centres alone then fix
# scale, yaw and translation but not the roll about the flight axis, and a
# 7-DoF fit lets GPS noise spin the model about that axis (a 300 m straight
# pass with metre-level GPS noise rolled ~30 deg in testing, moving ground 100 m
# off-track by tens of metres while the in-sample RMSE looked fine). The fix
# below levels the model on its ground plane first and fits only yaw, scale
# and translation, and reports leave-one-out error instead of in-sample.
# --------------------------------------------------------------------------


def track_shape(points: np.ndarray) -> dict[str, float]:
    """Singular-value shape of a point set: ``linearity = s2 / s1`` (0 = a line)."""
    arr = np.asarray(points, dtype=np.float64)
    centered = arr - arr.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    s1 = float(s[0]) if s[0] > 0 else 1.0
    return {
        "extent_m": float(np.ptp(centered @ np.linalg.svd(centered)[2][0])),
        "linearity": float(s[1] / s1),
        "planarity": float(s[2] / max(float(s[1]), 1e-12)),
    }


def estimate_up(
    points: np.ndarray,
    cameras: np.ndarray,
    *,
    iterations: int = 400,
    inlier_fraction_of_extent: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Ground-plane normal of a model, pointing towards its cameras.

    RANSAC plane on the sparse points (threshold 1 % of the scene extent);
    returns ``(unit_normal, inlier_fraction)``. Aerial scenes are dominated by
    ground, so the largest plane is the ground; the cameras decide its sign.
    """
    pts = np.asarray(points, dtype=np.float64)
    cams = np.asarray(cameras, dtype=np.float64)
    if len(pts) < 3:
        raise ReconstructionError("need at least 3 points to estimate the ground plane")
    rng = np.random.default_rng(seed)
    extent = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0
    threshold = inlier_fraction_of_extent * extent
    best_normal, best_count = np.array([0.0, 0.0, 1.0]), -1
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        count = int(np.sum(np.abs((pts - a) @ normal) < threshold))
        if count > best_count:
            best_normal, best_count = normal, count
            anchor = a
    # Refine on the inliers by PCA.
    inliers = pts[np.abs((pts - anchor) @ best_normal) < threshold]
    centered = inliers - inliers.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    if np.mean((cams - inliers.mean(axis=0)) @ normal) < 0:
        normal = -normal
    return normal, float(len(inliers) / len(pts))


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / float(np.dot(v, v)))


def solve_yaw_scale(src: np.ndarray, dst: np.ndarray, up_src: np.ndarray) -> SimilarityTransform:
    """4-DoF fit: level ``src`` so ``up_src`` maps to +z, then yaw, scale, translation.

    Scale and yaw come from the horizontal coordinates only (consumer GPS is
    worst vertically); the vertical offset is the mean residual.
    """
    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    if src_arr.shape != dst_arr.shape or src_arr.shape[0] < 2:
        raise ReconstructionError("yaw-scale fit needs >= 2 matching (N, 3) correspondences")
    level = _rotation_between(np.asarray(up_src, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
    lev = src_arr @ level.T
    p = lev[:, 0] + 1j * lev[:, 1]
    d = dst_arr[:, 0] + 1j * dst_arr[:, 1]
    pc, dc = p - p.mean(), d - d.mean()
    denom = float(np.sum(np.abs(pc) ** 2))
    if denom < 1e-12:
        raise ReconstructionError("camera centres do not spread horizontally")
    a = np.sum(np.conj(pc) * dc) / denom
    scale, theta = float(np.abs(a)), float(np.angle(a))
    yaw = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    rotation = yaw @ level
    t_xy = d.mean() - a * p.mean()
    t_z = float(np.mean(dst_arr[:, 2] - scale * lev[:, 2]))
    return SimilarityTransform(scale, rotation, np.array([t_xy.real, t_xy.imag, t_z]))


def leave_one_out(
    src: np.ndarray, dst: np.ndarray, solver, min_fit: int = 3
) -> dict[str, float] | None:
    """Held-out georeferencing error: refit without each correspondence, test on it."""
    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    n = len(src_arr)
    if n <= min_fit:
        return None
    idx = np.arange(n) if n <= 200 else np.random.default_rng(0).choice(n, 200, replace=False)
    residuals, scales = [], []
    for i in idx:
        keep = np.arange(n) != i
        transform = solver(src_arr[keep], dst_arr[keep])
        residuals.append(transform.apply(src_arr[i]) - dst_arr[i])
        scales.append(transform.scale)
    res = np.array(residuals)
    horizontal = np.linalg.norm(res[:, :2], axis=1)
    sc = np.array(scales)
    # Jackknife standard error of the scale: how much it moves when one GPS fix
    # is dropped, scaled up to the full-sample uncertainty.
    jack = float(np.sqrt((len(sc) - 1) / len(sc) * np.sum((sc - sc.mean()) ** 2)))
    return {
        "loo_rmse_horizontal_m": float(np.sqrt(np.mean(horizontal**2))),
        "loo_rmse_vertical_m": float(np.sqrt(np.mean(res[:, 2] ** 2))),
        "loo_n": int(len(idx)),
        "scale_relative_std": jack / float(sc.mean()),
    }


def solve_georef(
    model_centers: np.ndarray,
    gps_enu: np.ndarray,
    model_points: np.ndarray,
    *,
    mode: str = "auto",
    min_linearity: float = 0.15,
) -> tuple[SimilarityTransform, dict[str, Any]]:
    """Georeference a model from camera centres and GPS, robust to straight tracks.

    ``auto`` uses the full similarity only when the GPS track spans two
    dimensions (``linearity >= min_linearity``) *and* the similarity's implied
    vertical agrees with the model's ground plane within 10 deg; otherwise it
    levels on the ground plane and fits yaw, scale and translation.
    """
    src = np.asarray(model_centers, dtype=np.float64)
    dst = np.asarray(gps_enu, dtype=np.float64)
    shape = track_shape(dst)
    up, ground_fraction = estimate_up(model_points, src)
    info: dict[str, Any] = {"track": shape, "ground_inlier_fraction": ground_fraction}

    def yaw_solver(a: np.ndarray, b: np.ndarray) -> SimilarityTransform:
        return solve_yaw_scale(a, b, up)

    def sim_solver(a: np.ndarray, b: np.ndarray) -> SimilarityTransform:
        return solve_similarity(a, b, with_scale=True)

    chosen = mode
    if mode == "auto":
        chosen = "yaw-scale"
        if shape["linearity"] >= min_linearity and len(src) >= 3:
            sim = solve_similarity(src, dst, with_scale=True)
            tilt = float(np.degrees(np.arccos(np.clip((sim.rotation @ up)[2], -1.0, 1.0))))
            info["similarity_tilt_vs_ground_deg"] = tilt
            if tilt <= 10.0:
                chosen = "similarity"
    if chosen == "similarity":
        solver = sim_solver
    elif chosen == "yaw-scale":
        solver = yaw_solver
    elif chosen == "translation":

        def solver(a: np.ndarray, b: np.ndarray) -> SimilarityTransform:
            return SimilarityTransform(1.0, np.eye(3), b.mean(axis=0) - a.mean(axis=0))

    else:
        raise ReconstructionError(f"unknown align mode {mode!r}")
    transform = solver(src, dst)
    info["mode"] = chosen
    info["in_sample"] = gps_rmse(transform.apply(src), dst)
    loo = leave_one_out(src, dst, solver)
    if loo:
        info["held_out"] = loo
    return transform, info
