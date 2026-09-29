"""Texture atlas baked on the GPU from the keyframes, without UV unwrapping.

Open3D's UVAtlas took 14 s per 82k triangles on the CPU, rejected TSDF meshes
after decimation ("non-manifold") and could then crash the process. For a
mesh that is viewed and measured, not edited, a *triangle-soup* atlas is
enough: triangles are packed two per square cell of the texture, each texel
is mapped back to a point on its triangle, and that point is coloured from
the keyframe that sees the triangle most frontally and closely without being
occluded (a z-buffer of triangle centroids per view). Texels in a cell's
margin take the colour of the nearest point of their triangle, so bilinear
filtering in a viewer does not bleed neighbouring cells into each other.

Everything runs in torch on the GPU: a 600k-triangle mesh, 47 views and a
4096^2 texture take 4.6 s on the A100.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["View", "bake_soup_texture"]


@dataclass
class View:
    """A keyframe: intrinsics at its stored size (COLMAP convention), cam_from_world pose, RGB."""

    fx: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    image: torch.Tensor  # uint8 [H, W, 3] on the device
    k1: float = 0.0  # SIMPLE_RADIAL / RADIAL: at Jal Mahal's k1 = 0.12, 50 px at the image corners
    mask: torch.Tensor | None = None  # bool [H, W]: a burnt-in overlay this view must not texture from


def _project(pts: torch.Tensor, v: View) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = torch.as_tensor(v.rotation, dtype=pts.dtype, device=pts.device)
    t = torch.as_tensor(v.translation, dtype=pts.dtype, device=pts.device)
    pc = pts @ r.T + t
    z = pc[..., 2]
    x, y = pc[..., 0] / z, pc[..., 1] / z
    d = 1 + v.k1 * (x * x + y * y)
    return v.fx * d * x + v.cx - 0.5, v.fx * d * y + v.cy - 0.5, z


def _view_gains(views: list[View], cen: torch.Tensor, scores: torch.Tensor, *, sigma_n: float = 10.0,
                sigma_g: float = 0.1) -> tuple[torch.Tensor, dict]:  # fmt: skip
    """Per-view, per-channel gains that equalise exposure where views overlap -> ``([V, 3], stats)``.

    OpenCV's stitching ``GainCompensator`` / ``ChannelsCompensator`` objective,
    over triangles instead of panorama pixels: with ``N_ij`` triangles seen by
    views i and j and ``I_ij`` view i's mean colour over them, minimise
    ``sum N_ij ((g_i I_ij - g_j I_ji)^2 / sigma_n^2 + (1 - g_i)^2 / sigma_g^2)``.
    A drone's auto-exposure changes along a pass; the atlas blends the best
    views of each triangle, so uncorrected gains show as blotches and seams.
    """
    n_v = len(views)
    vis = scores > 0  # [V, F]
    cols = torch.zeros(n_v, cen.shape[0], 3, device=cen.device)
    for k, v in enumerate(views):
        sel = vis[k]
        if not bool(sel.any()):
            continue
        h, w = v.image.shape[:2]
        u, y, _ = _project(cen[sel], v)
        g = torch.stack([2 * u / (w - 1) - 1, 2 * y / (h - 1) - 1], -1)[None, None]
        img = v.image.permute(2, 0, 1)[None].float()
        cols[k, sel] = F.grid_sample(img, g, mode="bilinear", align_corners=True, padding_mode="border")[0, :, 0].T
    visf = vis.float()
    n_ij = visf @ visf.T  # [V, V] shared triangles
    # [V, V, 3]: view i's colour summed over the triangles it shares with j (one [V, F] @ [F, 3] per view)
    sums = torch.stack([(visf * visf[i]) @ cols[i] for i in range(n_v)])
    mean = sums / n_ij.clamp_min(1)[..., None]
    gains = torch.ones(n_v, 3, device=cen.device)
    # the normal equations in closed form: a Python loop over view pairs synced the GPU ~4 V^2 times per
    # channel, and a 47-view bake took 42.5 s (now 4.6 s)
    pair = ((n_ij >= 20) & ~torch.eye(n_v, dtype=torch.bool, device=cen.device)).double() * n_ij.double()
    for c in range(3):
        big_i = mean[..., c].double()
        a = -pair * big_i * big_i.T / sigma_n**2
        a.diagonal().copy_((pair * (big_i**2 / sigma_n**2 + 1 / sigma_g**2)).sum(1))
        b = pair.sum(1) / sigma_g**2
        live = a.diagonal() > 0
        if bool(live.any()):
            sol = torch.linalg.solve(a[live][:, live], b[live])
            gains[live, c] = sol.float().clamp(0.5, 2.0)
    # disagreement between overlapping views, before and after: mean |I_ij - I_ji| weighted by overlap
    mask = (n_ij >= 20) & ~torch.eye(n_v, dtype=torch.bool, device=cen.device)
    w = n_ij[mask]
    before = (mean - mean.transpose(0, 1)).abs().mean(-1)[mask]
    gm = mean * gains[:, None, :]
    after = (gm - gm.transpose(0, 1)).abs().mean(-1)[mask]
    stats = {"pairs": int(mask.sum()), "disagreement_before": round(float((before * w).sum() / w.sum().clamp_min(1)), 3) if bool(mask.any()) else None,
             "disagreement_after": round(float((after * w).sum() / w.sum().clamp_min(1)), 3) if bool(mask.any()) else None,
             "gain_range": [round(float(gains.min()), 3), round(float(gains.max()), 3)]}  # fmt: skip
    return gains, stats


@torch.inference_mode()
def soup_layout(n_f: int, size: int, device) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[no-untyped-def]
    """Two triangles per square cell of a ``size``^2 atlas -> (corner pixels [F, 3, 2], corner UVs [F, 3, 2], v up)."""
    cells = math.ceil(n_f / 2)
    per_side = math.ceil(math.sqrt(cells))
    cs = size // per_side
    if cs < 3:
        raise ValueError(f"{n_f} triangles do not fit a {size}^2 atlas")
    tid = torch.arange(n_f, device=device)
    cell_id, half = tid // 2, tid % 2
    ox = (cell_id % per_side).float() * cs
    oy = (cell_id // per_side).float() * cs
    m = 1.0  # texel margin kept free inside each cell
    lo, hi = m, cs - m
    # half 0: corners (lo,lo) (hi-1,lo) (lo,hi-1); half 1: (hi,hi) (lo+1,hi) (hi,lo+1) -- a 1-texel gap between them
    c0 = torch.tensor([[lo, lo], [hi - 1, lo], [lo, hi - 1]], device=device)
    c1 = torch.tensor([[hi, hi], [lo + 1, hi], [hi, lo + 1]], device=device)
    corners = torch.where(half[:, None, None] == 0, c0[None], c1[None]) + torch.stack([ox, oy], -1)[:, None]  # [F, 3, 2] px
    uv = corners / size
    return corners, torch.stack([uv[..., 0], 1.0 - uv[..., 1]], -1)  # OBJ/trimesh: v up


def soup_texels(corners: torch.Tensor, tri: torch.Tensor, size: int):  # type: ignore[no-untyped-def]
    """Every texel of the used cells of a ``soup_layout`` -> (flat index, triangle, in-use mask, barycentric
    weights (a, b, c), point on the triangle). Margin texels take the nearest point of their triangle, so
    bilinear filtering in a viewer does not bleed neighbouring cells into each other."""
    n_f = len(corners)
    cells = math.ceil(n_f / 2)
    per_side = math.ceil(math.sqrt(cells))
    cs = size // per_side
    dev = corners.device
    rows_used = math.ceil(cells / per_side) * cs
    ys, xs = torch.meshgrid(torch.arange(rows_used, device=dev), torch.arange(per_side * cs, device=dev), indexing="ij")
    px, py = xs.flatten().float() + 0.5, ys.flatten().float() + 0.5
    cid = (py.long() // cs) * per_side + (px.long() // cs)
    lx, ly = px - (px.long() // cs).float() * cs, py - (py.long() // cs).float() * cs
    t_of = cid * 2 + (lx + ly > cs).long()  # which half of the cell
    ok = t_of < n_f
    t_of = t_of.clamp(max=n_f - 1)
    a, b, c = corners[t_of, 0], corners[t_of, 1], corners[t_of, 2]
    p2 = torch.stack([px, py], -1)
    v0, v1, v2 = b - a, c - a, p2 - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp_min(1e-9)
    wb = ((d11 * d20 - d01 * d21) / den).clamp(0, 1)
    wc = ((d00 * d21 - d01 * d20) / den).clamp(0, 1)
    over = (wb + wc).clamp_min(1.0)  # margin texels: nearest point of the triangle
    wb, wc = wb / over, wc / over
    wa = 1 - wb - wc
    pts = wa[:, None] * tri[t_of, 0] + wb[:, None] * tri[t_of, 1] + wc[:, None] * tri[t_of, 2]
    return py.long() * size + px.long(), t_of, ok, (wa, wb, wc), pts


def bake_soup_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: list[View],
    *,
    size: int = 4096,
    zbuf: int = 256,
    fallback_rgb: np.ndarray | None = None,
    blend: int = 3,
    device: str = "cuda",
    gain: bool = True,
    occluders: str = "centroids",
    slope: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> ``(corner_uv [F, 3, 2] in OBJ convention (v up), albedo uint8 [size, size, 3], info)``.

    Args:
        zbuf: long side of the per-view centroid z-buffer used for occlusion.
        fallback_rgb: ``[N, 3]`` vertex colours for triangles no view sees.
        gain: equalise the views' exposure first (OpenCV stitching's gain-compensation objective).
    """
    dev = torch.device(device)
    vt = torch.as_tensor(vertices, dtype=torch.float32, device=dev)
    ft = torch.as_tensor(faces, dtype=torch.long, device=dev)
    n_f = len(ft)
    tri = vt[ft]  # [F, 3, 3]
    cen = tri.mean(1)
    nrm = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---- view scores per triangle: frontal, close, inside the frame, not occluded
    mid = (tri + tri.roll(-1, dims=1)) / 2  # edge midpoints
    samples = torch.cat([tri.reshape(-1, 3), mid.reshape(-1, 3), cen])  # [3F corners (per triangle), 3F midpoints, F]
    scores = torch.full((len(views), n_f), -1.0, device=dev)
    for k, v in enumerate(views):
        h, w = v.image.shape[:2]
        u, y, z = _project(cen, v)
        centre = -torch.as_tensor(v.rotation.T @ v.translation, dtype=torch.float32, device=dev)
        to_cam = centre - cen
        dist = to_cam.norm(dim=-1).clamp_min(1e-9)
        facing = (nrm * to_cam).sum(-1).abs() / dist  # TSDF normals may point either way
        inside = (z > 0) & (u >= 0) & (u <= w - 1) & (y >= 0) & (y <= h - 1)
        if v.mask is not None:  # a logo in this corner of the frame is not the surface behind it
            mh, mw = v.mask.shape
            mx = (u * (mw / w)).long().clamp(0, mw - 1)
            my = (y * (mh / h)).long().clamp(0, mh - 1)
            inside &= ~v.mask[my, mx]
        s = zbuf / max(h, w)
        zw, zh = max(1, int(w * s)), max(1, int(h * s))
        cx = (u * s).long().clamp(0, zw - 1)
        cy = (y * s).long().clamp(0, zh - 1)
        cell = cy * zw + cx
        # The occluders: the triangles' centroids ("samples": seven points of each, so that a large near triangle
        # covers every cell it spans -- measured, not better: experiments/texture_holdout.py).
        su, sy, sz = _project(samples if occluders == "samples" else torch.cat([tri.reshape(-1, 3), cen]), v)
        sin = (sz > 0) & (su >= 0) & (su <= w - 1) & (sy >= 0) & (sy <= h - 1)
        if occluders != "samples":  # the centroids alone occlude (the corners are projected for the slope only)
            sin[: 3 * n_f] = False
        scell = (sy * s).long().clamp(0, zh - 1) * zw + (su * s).long().clamp(0, zw - 1)
        zmin = torch.full((zh * zw,), float("inf"), device=dev)
        zmin.scatter_reduce_(0, scell[sin], sz[sin], reduce="amin")
        # A surface seen at a grazing angle rises across one z-buffer cell by more than a fixed 3 %: its farther
        # triangles then counted as hidden behind its nearer ones. The tolerance adds each triangle's own
        # depth slope times the cell's width. Held out on the largest model of every sample video (every 8th
        # keyframe rendered and compared with its photo), untextured triangles fell from a median 7.7 to 5.7 %
        # (at most 21 -> 15 %) and the PSNR moved by -0.07 to +0.24 dB.
        uc, yc, zc = su[: 3 * n_f].view(-1, 3), sy[: 3 * n_f].view(-1, 3), sz[: 3 * n_f].view(-1, 3)
        extent = torch.maximum(uc.amax(1) - uc.amin(1), yc.amax(1) - yc.amin(1)).clamp_min(0.5)
        rise = (zc.amax(1) - zc.amin(1)) / extent / s  # depth change across one cell along this surface
        visible = inside & (z <= zmin[cell] * 1.03 + (1.5 * rise if slope else 0.0))
        scores[k] = torch.where(visible, facing / dist, torch.full_like(dist, -1.0))
    # Blend the best ``blend`` views (weights by score, only views scoring >= half the best):
    # one view per triangle made water and other view-dependent surfaces a patchwork.
    top_s, top_v = scores.topk(min(blend, len(views)), dim=0)  # [k, F]
    best_view = torch.where(top_s[0] > 0, top_v[0], torch.full_like(top_v[0], -1))
    top_w = torch.where((top_s > 0) & (top_s >= 0.5 * top_s[:1]), top_s, torch.zeros_like(top_s))
    top_w = top_w / top_w.sum(0, keepdim=True).clamp_min(1e-12)
    gains, gain_stats = _view_gains(views, cen, scores) if gain and len(views) > 1 else (None, None)

    # ---- layout: two triangles per square cell; every texel of every used cell, as a point on its triangle
    corners, uv_obj = soup_layout(n_f, size, dev)
    flat_idx, t_of, ok, (wa, wb, wc), pts = soup_texels(corners, tri, size)
    albedo = torch.zeros(size * size, 3, dtype=torch.uint8, device=dev)
    view_of = best_view[t_of]
    col = torch.zeros(len(pts), 3, device=dev)
    for rank in range(top_v.shape[0]):
        v_r, w_r = top_v[rank][t_of], top_w[rank][t_of]
        for k, v in enumerate(views):
            sel = ok & (v_r == k) & (w_r > 0)
            if not bool(sel.any()):
                continue
            h, w = v.image.shape[:2]
            u, y, _ = _project(pts[sel], v)
            g = torch.stack([2 * u / (w - 1) - 1, 2 * y / (h - 1) - 1], -1)[None, None]
            img = v.image.permute(2, 0, 1)[None].float()
            sample = F.grid_sample(img, g, mode="bilinear", align_corners=True, padding_mode="border")[0, :, 0].T
            if gains is not None:
                sample = sample * gains[k]
            col[sel] += w_r[sel, None] * sample
    unseen = ok & (view_of < 0)
    if fallback_rgb is not None and bool(unseen.any()):
        fb = torch.as_tensor(fallback_rgb, dtype=torch.float32, device=dev)
        if fb.max() <= 1.0:
            fb = fb * 255
        col[unseen] = (wa[unseen, None] * fb[ft[t_of[unseen], 0]] + wb[unseen, None] * fb[ft[t_of[unseen], 1]]
                       + wc[unseen, None] * fb[ft[t_of[unseen], 2]])  # fmt: skip
    albedo[flat_idx[ok]] = col[ok].round().clamp(0, 255).to(torch.uint8)
    info = {"triangles": int(n_f), "cell_px": int(size // math.ceil(math.sqrt(math.ceil(n_f / 2)))), "views": len(views),
            "unseen_triangles": int((best_view < 0).sum()), "gain": gain_stats}  # fmt: skip
    return uv_obj.cpu().numpy(), albedo.view(size, size, 3).cpu().numpy(), info
