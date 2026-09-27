"""Batched two-view geometry on the GPU: homography, fundamental matrix, GRIC.

Every function takes a batch of correspondence sets padded to a common length,
``x, y: [B, N, 2]`` with per-correspondence weights ``w: [B, N]`` (0 = padding
or rejected), so hundreds of keyframe pairs are fitted in one call.

The question these answer is the one keyframe selection has to settle: does a
pair of views see *3D structure*, or only a motion a homography explains
(pure rotation, a planar scene, or a scene so far away it is effectively
planar)? Torr's Geometric Robust Information Criterion compares the two
models with the complexity each spends, and the residual a robust homography
leaves behind is the rotation-compensated parallax.

References:
    P. H. S. Torr, "Geometric Motion Segmentation and Model Selection",
    Phil. Trans. R. Soc. A, 1998.
    R. Hartley, "In Defense of the Eight-Point Algorithm", PAMI 1997.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "TwoViewFit",
    "fit_fundamental",
    "fit_homography",
    "gric",
    "sampson_sq",
    "transfer_sq",
    "two_view_analysis",
]


def _normalizer(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Hartley normalization ``T`` (``[B, 3, 3]``): weighted centroid to 0, mean radius sqrt 2."""
    wsum = w.sum(-1, keepdim=True).clamp_min(1e-9)
    centroid = (p * w[..., None]).sum(-2) / wsum
    radius = ((p - centroid[:, None]).norm(dim=-1) * w).sum(-1, keepdim=True) / wsum
    scale = math.sqrt(2.0) / radius.clamp_min(1e-9)
    t = torch.zeros(p.shape[0], 3, 3, dtype=p.dtype, device=p.device)
    t[:, 0, 0] = scale[:, 0]
    t[:, 1, 1] = scale[:, 0]
    t[:, 0, 2] = -scale[:, 0] * centroid[:, 0]
    t[:, 1, 2] = -scale[:, 0] * centroid[:, 1]
    t[:, 2, 2] = 1.0
    return t


def _apply(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply ``[B, 3, 3]`` projective maps to ``[B, N, 2]`` points."""
    hom = torch.cat([p, torch.ones_like(p[..., :1])], dim=-1) @ t.transpose(-1, -2)
    return hom[..., :2] / hom[..., 2:3].where(
        hom[..., 2:3].abs() > 1e-12, torch.full_like(hom[..., 2:3], 1e-12)
    )


def _smallest_eigvec(rows: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted least-squares null vector of ``rows [B, R, 9]`` (weights ``[B, R]``)."""
    m = (rows * w[..., None]).transpose(-1, -2) @ rows
    _, vecs = torch.linalg.eigh(m.double())
    return vecs[..., 0].to(rows.dtype)


def transfer_sq(h: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared one-sided transfer error ``|y - H x|^2`` in pixels^2, ``[B, N]``."""
    return (_apply(h, x) - y).pow(2).sum(-1)


def sampson_sq(f: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Sampson distance of ``y^T F x = 0`` in pixels^2, ``[B, N]``."""
    xh = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    yh = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)
    fx = xh @ f.transpose(-1, -2)
    fty = yh @ f
    num = (yh * fx).sum(-1).pow(2)
    den = fx[..., 0] ** 2 + fx[..., 1] ** 2 + fty[..., 0] ** 2 + fty[..., 1] ** 2
    return num / den.clamp_min(1e-12)


def _irls_weights(err_sq: torch.Tensor, w0: torch.Tensor, sigma: float) -> torch.Tensor:
    """Cauchy weights at scale ``max(sigma, 1.4826 * MAD-like median)``."""
    r = err_sq.clamp_min(0).sqrt()
    masked = torch.where(w0 > 0, r, torch.full_like(r, float("nan")))
    med = torch.nanmedian(masked, dim=-1, keepdim=True).values.nan_to_num(sigma)
    scale = (1.4826 * med).clamp_min(sigma) * 2.5
    return w0 / (1.0 + (r / scale) ** 2)


def fit_homography(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, *, iters: int = 6, sigma: float = 1.0
) -> torch.Tensor:
    """Robust normalized-DLT homography ``x -> y``, IRLS with Cauchy weights. ``[B, 3, 3]``."""
    tx, ty = _normalizer(x, w), _normalizer(y, w)
    xn, yn = _apply(tx, x), _apply(ty, y)
    u, v = xn[..., 0], xn[..., 1]
    up, vp = yn[..., 0], yn[..., 1]
    one, zero = torch.ones_like(u), torch.zeros_like(u)
    r1 = torch.stack([zero, zero, zero, -u, -v, -one, vp * u, vp * v, vp], dim=-1)
    r2 = torch.stack([u, v, one, zero, zero, zero, -up * u, -up * v, -up], dim=-1)
    rows = torch.cat([r1, r2], dim=1)
    weights = w
    h = torch.eye(3, dtype=x.dtype, device=x.device).expand(x.shape[0], 3, 3)
    for _ in range(max(1, iters)):
        hn = _smallest_eigvec(rows, torch.cat([weights, weights], dim=1)).reshape(-1, 3, 3)
        h = torch.linalg.solve(ty, hn @ tx)
        h = h / h[:, 2:3, 2:3].where(h[:, 2:3, 2:3].abs() > 1e-12, torch.ones_like(h[:, 2:3, 2:3]))
        weights = _irls_weights(transfer_sq(h, x, y), w, sigma)
    return h


def fit_fundamental(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, *, iters: int = 6, sigma: float = 1.0
) -> torch.Tensor:
    """Robust normalized eight-point fundamental matrix (``y^T F x = 0``), rank 2. ``[B, 3, 3]``."""
    tx, ty = _normalizer(x, w), _normalizer(y, w)
    xn, yn = _apply(tx, x), _apply(ty, y)
    u, v = xn[..., 0], xn[..., 1]
    up, vp = yn[..., 0], yn[..., 1]
    rows = torch.stack([up * u, up * v, up, vp * u, vp * v, vp, u, v, torch.ones_like(u)], dim=-1)
    weights = w
    f = torch.zeros(x.shape[0], 3, 3, dtype=x.dtype, device=x.device)
    for _ in range(max(1, iters)):
        fn = _smallest_eigvec(rows, weights).reshape(-1, 3, 3)
        uu, ss, vv = torch.linalg.svd(fn.double())
        ss = ss.clone()
        ss[:, 2] = 0.0
        fn = (uu @ torch.diag_embed(ss) @ vv).to(x.dtype)
        f = ty.transpose(-1, -2) @ fn @ tx
        f = f / f.flatten(1).norm(dim=-1).clamp_min(1e-12)[:, None, None]
        weights = _irls_weights(sampson_sq(f, x, y), w, sigma)
    return f


def gric(
    err_sq: torch.Tensor, w: torch.Tensor, *, sigma: float | torch.Tensor, d: int, k: int
) -> torch.Tensor:
    """Torr's GRIC for a model with manifold dimension ``d`` and ``k`` parameters.

    ``GRIC = sum_i min(e_i^2 / sigma^2, lambda3 (r - d)) + lambda1 d n + lambda2 k``
    with ``r = 4`` (two 2D views), ``lambda1 = ln r``, ``lambda2 = ln(r n)``,
    ``lambda3 = 2``. ``e_i`` must be the *geometric* (reprojection) error of the
    model so the two models are compared on the same footing, and ``sigma``
    (scalar or ``[B]``) the per-coordinate noise. Lower is better. ``w > 0``
    marks correspondences that count.
    """
    r = 4.0
    valid = (w > 0).to(err_sq.dtype)
    n = valid.sum(-1)
    sig = torch.as_tensor(sigma, dtype=err_sq.dtype, device=err_sq.device)
    sig = sig[..., None] if sig.ndim == 1 else sig
    rho = torch.minimum(err_sq / sig**2, torch.full_like(err_sq, 2.0 * (r - d)))
    return (rho * valid).sum(-1) + math.log(r) * d * n + torch.log((r * n).clamp_min(1.0)) * k


# Median |e| of 2D isotropic Gaussian noise with per-coordinate sigma * sqrt(2)
# (noise in both views): sqrt(2) * sqrt(2 ln 2) * sigma. A transfer residual this
# size is what a homography leaves on pure noise.
_NOISE_TRANSFER_MEDIAN = math.sqrt(2.0) * math.sqrt(2.0 * math.log(2.0))


@dataclass
class TwoViewFit:
    """Per-pair results of :func:`two_view_analysis` (all tensors ``[B]`` unless noted)."""

    homography: torch.Tensor  # [B, 3, 3]
    fundamental: torch.Tensor  # [B, 3, 3]
    num_valid: torch.Tensor
    sigma_px: torch.Tensor  # robust noise scale from F's residuals
    parallax_px: torch.Tensor  # median |y - Hx| over valid correspondences
    parallax_deg: torch.Tensor  # atan(parallax_px / focal_px)
    parallax_snr: torch.Tensor  # parallax_px / what pure noise leaves (1 = no parallax)
    gric_h: torch.Tensor
    gric_f: torch.Tensor
    prefers_3d: torch.Tensor  # bool: GRIC(F) < GRIC(H)
    epipolar_rms_px: torch.Tensor  # sqrt(median Sampson^2): how well F fits


def two_view_analysis(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    focal_px: float,
    sigma: float | None = None,
    sigma_floor: float = 0.05,
    iters: int = 6,
) -> TwoViewFit:
    """Fit H and F to every pair, compare them by GRIC and measure parallax.

    Args:
        x, y: ``[B, N, 2]`` pixel correspondences (view 1 -> view 2).
        w: ``[B, N]`` 1 for a real correspondence, 0 for padding.
        focal_px: focal length in the same pixels, to turn parallax into an angle.
        sigma: per-coordinate correspondence noise. ``None`` estimates it per
            pair from F's residuals (``1.4826 * median sqrt(Sampson)``), which
            are pure noise whenever F is the right model -- and F is right for
            degenerate motion too, since H-consistent pairs also satisfy some F.
    """
    x = x.float()
    y = y.float()
    w = w.float()
    fit_sigma = 1.0 if sigma is None else sigma
    h = fit_homography(x, y, w, iters=iters, sigma=fit_sigma)
    f = fit_fundamental(x, y, w, iters=iters, sigma=fit_sigma)
    eh = transfer_sq(h, x, y)
    ef = sampson_sq(f, x, y)
    valid = w > 0
    nan = torch.full_like(eh, float("nan"))
    parallax = torch.nanmedian(torch.where(valid, eh.sqrt(), nan), dim=-1).values
    epi = torch.nanmedian(torch.where(valid, ef, nan), dim=-1).values.sqrt()
    if sigma is None:
        sig = (
            1.4826 * torch.nanmedian(torch.where(valid, ef.sqrt(), nan), dim=-1).values
        ).clamp_min(sigma_floor)
    else:
        sig = torch.full_like(parallax, float(sigma))
    # Both models scored on geometric error: Sampson already approximates it
    # for F; for H the transfer error carries both views' noise, so halve it.
    g_h = gric(eh / 2.0, w, sigma=sig, d=2, k=8)
    g_f = gric(ef, w, sigma=sig, d=3, k=7)
    return TwoViewFit(
        homography=h,
        fundamental=f,
        num_valid=valid.sum(-1),
        sigma_px=sig,
        parallax_px=parallax,
        parallax_deg=torch.rad2deg(torch.atan(parallax / focal_px)),
        parallax_snr=parallax / (_NOISE_TRANSFER_MEDIAN * sig),
        gric_h=g_h,
        gric_f=g_f,
        prefers_3d=g_f < g_h,
        epipolar_rms_px=epi,
    )
