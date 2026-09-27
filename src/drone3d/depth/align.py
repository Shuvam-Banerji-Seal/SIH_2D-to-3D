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
    "MonotoneMap",
    "cross_validate",
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


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the non-decreasing fit to ``y``."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi))
        wts.append(float(wi))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, n2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, n1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            sizes.append(n1 + n2)
    return np.repeat(vals, sizes)


class MonotoneMap:
    """Monotone map ``pred -> log depth`` from binned medians (isotonic), linear ends.

    Affine-invariant depth is affine only over the depth range its training
    data covered; at landscape scale the far field comes out compressed. A
    monotone calibration keeps the prediction's ordering but lets the mapping
    bend. Knots are medians of ``log z`` in quantile bins of ``pred`` (robust to
    outliers), made non-decreasing by weighted PAV; outside the knots the map
    continues with the robust affine slope.
    """

    def __init__(self, pred: np.ndarray, z: np.ndarray, *, bins: int = 24, slope: float = 1.0) -> None:
        p, lz = np.asarray(pred, dtype=np.float64), np.log(np.asarray(z, dtype=np.float64))
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        xs, ys, ws = [], [], []
        for k in range(len(edges) - 1):
            m = which == k
            if m.sum() >= 3:
                xs.append(float(np.median(p[m])))
                ys.append(float(np.median(lz[m])))
                ws.append(float(m.sum()))
        if len(xs) < 2:
            raise ValueError("not enough spread in the prediction to calibrate")
        self.x = np.asarray(xs)
        self.y = _pav(np.asarray(ys), np.asarray(ws))
        self.slope = max(slope, 1e-3)

    def __call__(self, pred):  # type: ignore[no-untyped-def]
        """Log depth for ``pred`` (numpy array or torch tensor)."""
        try:
            import torch

            if isinstance(pred, torch.Tensor):
                x = torch.as_tensor(self.x, dtype=pred.dtype, device=pred.device)
                y = torch.as_tensor(self.y, dtype=pred.dtype, device=pred.device)
                idx = torch.clamp(torch.searchsorted(x, pred.contiguous()), 1, len(self.x) - 1)
                x0, x1, y0, y1 = x[idx - 1], x[idx], y[idx - 1], y[idx]
                t = (pred - x0) / torch.clamp(x1 - x0, min=1e-9)
                inner = y0 + t * (y1 - y0)
                lo = y[0] + self.slope * (pred - x[0])
                hi = y[-1] + self.slope * (pred - x[-1])
                return torch.where(pred < x[0], lo, torch.where(pred > x[-1], hi, inner))
        except ImportError:  # pragma: no cover
            pass
        p = np.asarray(pred, dtype=np.float64)
        out = np.interp(p, self.x, self.y)
        out = np.where(p < self.x[0], self.y[0] + self.slope * (p - self.x[0]), out)
        return np.where(p > self.x[-1], self.y[-1] + self.slope * (p - self.x[-1]), out)


def _scores(d: np.ndarray, z: np.ndarray) -> dict[str, float]:
    rel = np.abs(d - z) / z
    return {"abs_rel_median": float(np.median(rel)), "delta1": float(np.mean(np.maximum(d / z, z / d) < 1.25))}


def cross_validate(pred: np.ndarray, z: np.ndarray, folds: int = 5, seed: int = 0) -> dict[str, dict[str, float]] | None:
    """Held-out-point comparison of affine vs monotone calibration."""
    ok = np.isfinite(pred) & np.isfinite(z) & (z > 0)
    p, zz = pred[ok], z[ok]
    if len(p) < 10 * folds:
        return None
    order = np.random.default_rng(seed).permutation(len(p))
    parts = np.array_split(order, folds)
    out: dict[str, list] = {"affine": [], "monotone": []}
    for k in range(folds):
        test = parts[k]
        train = np.concatenate([parts[j] for j in range(folds) if j != k])
        fit = fit_log_affine(p[train], zz[train])
        if fit is None or fit.a <= 0:
            return None
        out["affine"].append(np.exp(fit.a * p[test] + fit.b))
        try:
            mono = MonotoneMap(p[train], zz[train], slope=fit.a)
        except ValueError:
            return None
        out["monotone"].append(np.exp(mono(p[test])))
    z_test = np.concatenate([zz[parts[k]] for k in range(folds)])
    return {name: _scores(np.concatenate(v), z_test) for name, v in out.items()}
