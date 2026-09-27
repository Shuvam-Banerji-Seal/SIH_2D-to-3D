"""Edge-aware refinement of fused depth maps with OpenCV 5's ximgproc filters, before TSDF fusion.

A fused depth map mixes two sources: flow triangulation (accurate where it
exists) and the calibrated monocular fill (complete but smoother and less
exact). Both are noisy at object boundaries, where a pixel's depth belongs to
one surface or the other; the TSDF averages that noise into fuzzy edges and
floaters. Filtering *inverse* depth (linear in the image) under the keyframe's
own colours keeps edges where the photo has them:

- ``guided``: He et al.'s guided filter (``cv2.ximgproc.guidedFilter``), masked
  so empty pixels neither contribute nor get filled;
- ``fgs``: the Fast Global Smoother (``cv2.ximgproc.fastGlobalSmootherFilter``,
  an edge-aware weighted least squares), with a confidence per pixel --
  triangulated depth trusted, the monocular fill less -- so the fill bends
  towards the measured geometry across a surface. (The Fast Bilateral Solver
  would be the natural choice; the pip build of OpenCV 5 lacks Eigen for it.)

Sky and pixels without depth stay empty either way.
"""

from __future__ import annotations

import numpy as np

__all__ = ["refine_depths"]


def refine_depths(depths: list[np.ndarray], rgb: np.ndarray, tri_depths: list[np.ndarray], *, method: str = "none",
                  radius: int = 4, eps: float = 0.02, fill_confidence: float = 0.35, lam: float = 32.0) -> list[np.ndarray]:  # fmt: skip
    """``depths`` (float32 [H, W], 0 = none) refined under ``rgb`` (uint8 [N, H, W, 3]); ``method``: none | guided | fgs."""
    if method == "none":
        return depths
    import cv2

    if method not in ("guided", "fgs"):
        raise ValueError(f"unknown depth refinement {method!r} (none | guided | fgs)")
    out = []
    for d, img, tri in zip(depths, rgb, tri_depths, strict=True):
        valid = d > 0
        if valid.sum() < 64:
            out.append(d)
            continue
        guide = np.ascontiguousarray(img)
        inv = np.where(valid, 1.0 / np.maximum(d, 1e-9), 0.0).astype(np.float32)
        scale = float(np.percentile(inv[valid], 99)) or 1.0  # filters behave in [0, 1]
        inv /= scale
        if method == "guided":
            m = valid.astype(np.float32)
            e = (eps * 255.0) ** 2  # the guide is 8-bit
            num = cv2.ximgproc.guidedFilter(guide, inv * m, radius, e)
            den = cv2.ximgproc.guidedFilter(guide, m, radius, e)
            ref = np.where(den > 0.25, num / np.maximum(den, 1e-6), 0.0)
        else:
            conf = np.where(tri > 0, 1.0, np.where(valid, fill_confidence, 0.0)).astype(np.float32)
            num = cv2.ximgproc.fastGlobalSmootherFilter(guide, inv * conf, lam, 8.0)
            den = cv2.ximgproc.fastGlobalSmootherFilter(guide, conf, lam, 8.0)
            ref = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0)
        ref = ref.astype(np.float32) * scale
        out.append(
            np.where(valid & (ref > 0), 1.0 / np.maximum(ref, 1e-12), 0.0).astype(np.float32)
        )
    return out
