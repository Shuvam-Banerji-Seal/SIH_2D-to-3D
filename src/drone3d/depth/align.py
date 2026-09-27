"""Align affine-invariant monocular depth to SfM, and score it against SfM.

Marigold v2 predicts ``pred`` with ``log d = a * pred + b`` for unknown
``a, b`` per image. Every registered image has SfM tie points with a depth
``z`` in its camera frame, so ``a, b`` follow from a robust line fit in log
space. The same points then measure how well the prior agrees with
multi-view geometry (AbsRel, delta < 1.25), which is the number that decides
how much the splat trainer should trust it.

Depth maps are written the way spirula-studio reads them: 16-bit PNG, linear
z-depth, ``0`` = no data, scaled so the 99.9th percentile maps to 65535 (its
depth loss is a Pearson correlation, so the per-image scale is irrelevant but
the log -> linear mapping, set by ``a``, is not).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "AlignResult",
    "DepthObservations",
    "fit_log_affine",
    "sample_at",
    "sfm_depth_observations",
    "write_depth_png",
]


@dataclass
class DepthObservations:
    """SfM tie points seen by one image: pixel ``uv`` (COLMAP convention) and depth ``z``."""

    name: str
    width: int
    height: int
    uv: np.ndarray  # [N, 2]
    z: np.ndarray  # [N]
    cam_from_world: np.ndarray  # [3, 4]
    params: np.ndarray
    model: str


def sfm_depth_observations(model_dir: str | Path) -> dict[str, DepthObservations]:
    """Per-image tie points and depths from a COLMAP model (``.bin`` or ``.txt``)."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    out: dict[str, DepthObservations] = {}
    for image in rec.images.values():
        if not image.has_pose:
            continue
        pts = [p for p in image.points2D if p.has_point3D()]
        if not pts:
            continue
        uv = np.array([p.xy for p in pts], dtype=np.float64)
        xyz = np.array([rec.points3D[p.point3D_id].xyz for p in pts], dtype=np.float64)
        t = image.cam_from_world().matrix()
        z = xyz @ t[:, :3].T[:, 2] + t[2, 3]
        keep = z > 0
        cam = rec.cameras[image.camera_id]
        out[image.name] = DepthObservations(
            name=image.name,
            width=int(cam.width),
            height=int(cam.height),
            uv=uv[keep],
            z=z[keep],
            cam_from_world=t,
            params=np.asarray(cam.params, dtype=np.float64),
            model=str(cam.model).split(".")[-1],
        )
    return out


def sample_at(field: torch.Tensor, uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinearly sample ``field [C, h, w]`` at image pixels ``uv`` of a ``width x height`` image.

    COLMAP puts pixel centres at half-integers, so ``u / width`` is already
    the normalized coordinate of ``align_corners=False``.
    """
    g = torch.as_tensor(uv, dtype=torch.float32, device=field.device)
    g = torch.stack([2 * g[:, 0] / width - 1, 2 * g[:, 1] / height - 1], -1)
    out = F.grid_sample(field[None].float(), g[None, None], mode="bilinear", align_corners=False)
    return out[0, :, 0].T.cpu().numpy().squeeze(-1)


@dataclass
class AlignResult:
    a: float
    b: float
    num_points: int
    inlier_fraction: float
    abs_rel: float  # mean |d - z| / z over all points after alignment
    abs_rel_median: float
    delta1: float  # share with max(d/z, z/d) < 1.25
    log_rmse: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(v), 5) for k, v in asdict(self).items()}


def fit_log_affine(pred: np.ndarray, z: np.ndarray, *, iters: int = 12) -> AlignResult | None:
    """Robust fit of ``log z = a * pred + b`` (IRLS, Cauchy weights at 2.5 x robust scale)."""
    ok = np.isfinite(pred) & np.isfinite(z) & (z > 0)
    if ok.sum() < 8:
        return None
    p, lz = pred[ok].astype(np.float64), np.log(z[ok])
    w = np.ones_like(p)
    a = b = 0.0
    for _ in range(iters):
        sw = w.sum()
        mp, mz = (w * p).sum() / sw, (w * lz).sum() / sw
        var = (w * (p - mp) ** 2).sum()
        if var <= 1e-12:
            return None
        a = (w * (p - mp) * (lz - mz)).sum() / var
        b = mz - a * mp
        r = lz - (a * p + b)
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3) * 2.5
        w = 1.0 / (1.0 + (r / scale) ** 2)
    r = lz - (a * p + b)
    d = np.exp(a * p + b)
    ratio = np.maximum(d / z[ok], z[ok] / d)
    scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3)
    return AlignResult(
        a=float(a),
        b=float(b),
        num_points=int(ok.sum()),
        inlier_fraction=float(np.mean(np.abs(r) < 3 * scale)),
        abs_rel=float(np.mean(np.abs(d - z[ok]) / z[ok])),
        abs_rel_median=float(np.median(np.abs(d - z[ok]) / z[ok])),
        delta1=float(np.mean(ratio < 1.25)),
        log_rmse=float(np.sqrt(np.mean(r**2))),
    )


def write_depth_png(path: str | Path, depth: np.ndarray, valid: np.ndarray | None = None) -> None:
    """Write linear depth as spirula's 16-bit PNG (0 = no data, 99.9 % -> 65535)."""
    import cv2

    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= valid
    out = np.zeros(d.shape, dtype=np.uint16)
    if ok.any():
        scale = max(float(np.quantile(d[ok], 0.999)), 1e-9)
        out[ok] = np.clip(np.rint(d[ok] / scale * 65535.0), 1, 65535).astype(np.uint16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"could not write {path}")
