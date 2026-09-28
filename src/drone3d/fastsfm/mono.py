"""Monocular depth to complete the triangulated depth maps.

A single pass gives little parallax on distant or textureless surfaces (the
far shore, hills, water): flow triangulation leaves them empty. Depth
Anything V2 predicts relative disparity everywhere; per image it is mapped to
metric-consistent depth by a monotone calibration fitted to that image's own
triangulated pixels (``drone3d.depth.align.MonotoneMap``, the calibration that
halved held-out error for Marigold at landscape scale), and fills only the
pixels triangulation could not reach. Pixels mapped beyond ``far_factor`` x
the farthest calibrated depth (sky) stay empty.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["MonoDepth", "calibrate_fill", "load_mono"]

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def load_mono(model: str = "depth-anything/Depth-Anything-V2-Large-hf", device: str = "cuda") -> torch.nn.Module:
    """A transformers depth model in fp16, eval mode, on ``device`` (weights from ``/store/huggingface``)."""
    from transformers import AutoModelForDepthEstimation

    os.environ.setdefault("HF_HUB_CACHE", "/store/huggingface")
    return AutoModelForDepthEstimation.from_pretrained(model, dtype=torch.float16).to(device).eval()


class MonoDepth:
    """Batched Depth Anything V2 (transformers) in fp16 -> relative disparity at the frames' size."""

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Large-hf", *, device: str = "cuda",
                 long_side: int = 700, batch: int = 16, net: torch.nn.Module | None = None) -> None:  # fmt: skip
        self.net = net if net is not None else load_mono(model, device)
        self.device, self.long_side, self.batch, self.name = torch.device(device), long_side, batch, model

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, H, W, 3]`` -> float32 ``[B, H, W]`` relative disparity (larger = nearer)."""
        b, h, w, _ = frames.shape
        s = self.long_side / max(h, w)
        th, tw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
        out = torch.empty(b, h, w, device=self.device)
        mean, std = _MEAN.to(self.device)[None, :, None, None], _STD.to(self.device)[None, :, None, None]
        for i in range(0, b, self.batch):
            x = frames[i : i + self.batch].to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
            x = ((x - mean) / std).half()
            pred = self.net(pixel_values=x).predicted_depth.float()  # [b, th, tw]
            out[i : i + len(x)] = F.interpolate(pred[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        return out


WALL_RATIO = 0.03  # triangulated vs prior log-depth spread below which a view's geometry is rejected (see below)


def calibrate_fill(disparity: np.ndarray, tri_depth: np.ndarray, *, far_factor: float = 3.0,
                   min_samples: int = 400, sky_rel: float = 0.005) -> tuple[np.ndarray, dict]:  # fmt: skip
    """Fill ``tri_depth``'s empty pixels (0) from ``disparity`` calibrated on its filled ones.

    Depth Anything V2 was trained with sky at zero disparity (it returns exactly 0 there on
    our footage, against ~130 on the palace): pixels below ``sky_rel`` x the image's maximum
    disparity are sky -- never filled, and stray triangulated depths there are dropped.

    Returns ``(depth, info)``; ``info["status"]`` is ``filled``, or why the image
    was left as triangulated (too few samples, prediction not monotone in depth).
    """
    from drone3d.depth.align import MonotoneMap, fit_log_affine

    sky = disparity <= sky_rel * max(float(disparity.max()), 1e-6)
    tri_depth = np.where(sky, 0.0, tri_depth).astype(np.float32)
    p = -np.log(np.maximum(disparity, 1e-6))  # increases with distance
    have = tri_depth > 0
    info: dict = {"triangulated": round(float(have.mean()), 4), "sky": round(float(sky.mean()), 4)}
    if have.sum() < min_samples:
        return tri_depth, {**info, "status": "too-few-samples"}
    p_at, z = p[have].astype(np.float64), tri_depth[have].astype(np.float64)
    # A view whose triangulated depth is a wall where the prior sees depth has wrong geometry, not a flat
    # scene: a collapsed SfM model (Qutub Minar's hovering keyframes) put every pixel within 1-2 % of one
    # depth, and filling it built walls that crashed Open3D's extraction. Healthy views spread their
    # log-depth at least 0.44 x as much as the prior's (Jal Mahal, Kinbane, Cristo Redentor, Hagia Sophia);
    # that model, 0.01 x. Nadir views of flat ground are safe: there the prior is flat too.
    p_spread = float(np.quantile(p_at, 0.95) - np.quantile(p_at, 0.05))
    z_spread = float(np.quantile(np.log(z), 0.95) - np.quantile(np.log(z), 0.05))
    if p_spread > 0.3 and z_spread < WALL_RATIO * p_spread:
        return np.zeros_like(tri_depth), {**info, "status": "depth-contradicts-prior",
                                          "spread_ratio": round(z_spread / p_spread, 4)}  # fmt: skip
    fit = fit_log_affine(p_at, z)
    if fit is None or fit.a <= 0:
        return tri_depth, {**info, "status": "prior-not-monotone"}
    try:
        calib = MonotoneMap(p_at, z, slope=fit.a)
    except ValueError:
        return tri_depth, {**info, "status": "calibration-failed"}
    mono = np.exp(calib(p))  # numpy: torch's CPU path is 24-thread and ~10x slower on a busy host
    far = far_factor * float(np.quantile(z, 0.99))
    fill = ~have & ~sky & (mono > 0) & (mono < far)
    depth = np.where(have, tri_depth, np.where(fill, mono, 0.0)).astype(np.float32)
    rel = np.abs(np.exp(calib(p_at)) - z) / z
    info.update(status="filled", filled=round(float(fill.mean()), 4), in_sample_abs_rel=round(float(np.median(rel)), 4))
    return depth, info
