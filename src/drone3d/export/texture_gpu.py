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

Everything runs in torch on the GPU: a 250k-triangle mesh, 12 views and a
4096^2 texture take a few seconds.
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
    """A keyframe: pinhole intrinsics at its stored size (COLMAP convention), cam_from_world pose, RGB."""

    fx: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    image: torch.Tensor  # uint8 [H, W, 3] on the device


def _project(pts: torch.Tensor, v: View) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = torch.as_tensor(v.rotation, dtype=pts.dtype, device=pts.device)
    t = torch.as_tensor(v.translation, dtype=pts.dtype, device=pts.device)
    pc = pts @ r.T + t
    z = pc[..., 2]
    u = v.fx * pc[..., 0] / z + v.cx - 0.5
    w = v.fx * pc[..., 1] / z + v.cy - 0.5
    return u, w, z


@torch.inference_mode()
def bake_soup_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: list[View],
    *,
    size: int = 4096,
    zbuf: int = 256,
    fallback_rgb: np.ndarray | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> ``(corner_uv [F, 3, 2] in OBJ convention (v up), albedo uint8 [size, size, 3], info)``.

    Args:
        zbuf: long side of the per-view centroid z-buffer used for occlusion.
        fallback_rgb: ``[N, 3]`` vertex colours for triangles no view sees.
    """
    dev = torch.device(device)
    vt = torch.as_tensor(vertices, dtype=torch.float32, device=dev)
    ft = torch.as_tensor(faces, dtype=torch.long, device=dev)
    n_f = len(ft)
    tri = vt[ft]  # [F, 3, 3]
    cen = tri.mean(1)
    nrm = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---- best view per triangle: frontal, close, inside the frame, not occluded
    best_score = torch.full((n_f,), -1.0, device=dev)
    best_view = torch.full((n_f,), -1, dtype=torch.long, device=dev)
    for k, v in enumerate(views):
        h, w = v.image.shape[:2]
        u, y, z = _project(cen, v)
        centre = -torch.as_tensor(v.rotation.T @ v.translation, dtype=torch.float32, device=dev)
        to_cam = centre - cen
        dist = to_cam.norm(dim=-1).clamp_min(1e-9)
        facing = (nrm * to_cam).sum(-1).abs() / dist  # TSDF normals may point either way
        inside = (z > 0) & (u >= 0) & (u <= w - 1) & (y >= 0) & (y <= h - 1)
        s = zbuf / max(h, w)
        zw, zh = max(1, int(w * s)), max(1, int(h * s))
        cx = (u * s).long().clamp(0, zw - 1)
        cy = (y * s).long().clamp(0, zh - 1)
        cell = cy * zw + cx
        zmin = torch.full((zh * zw,), float("inf"), device=dev)
        zmin.scatter_reduce_(0, cell[inside], z[inside], reduce="amin")
        visible = inside & (z <= zmin[cell] * 1.03)
        score = torch.where(visible, facing / dist, torch.full_like(dist, -1.0))
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        best_view = torch.where(better, torch.full_like(best_view, k), best_view)

    # ---- layout: two triangles per square cell
    cells = math.ceil(n_f / 2)
    per_side = math.ceil(math.sqrt(cells))
    cs = size // per_side
    if cs < 3:
        raise ValueError(f"{n_f} triangles do not fit a {size}^2 atlas")
    tid = torch.arange(n_f, device=dev)
    cell_id, half = tid // 2, tid % 2
    ox = (cell_id % per_side).float() * cs
    oy = (cell_id // per_side).float() * cs
    m = 1.0  # texel margin kept free inside each cell
    lo, hi = m, cs - m
    # half 0: corners (lo,lo) (hi-1,lo) (lo,hi-1); half 1: (hi,hi) (lo+1,hi) (hi,lo+1) -- a 1-texel gap between them
    c0 = torch.tensor([[lo, lo], [hi - 1, lo], [lo, hi - 1]], device=dev)
    c1 = torch.tensor([[hi, hi], [lo + 1, hi], [hi, lo + 1]], device=dev)
    corners = torch.where(half[:, None, None] == 0, c0[None], c1[None]) + torch.stack([ox, oy], -1)[:, None]  # [F, 3, 2] px
    uv = corners / size
    uv_obj = torch.stack([uv[..., 0], 1.0 - uv[..., 1]], -1)  # OBJ/trimesh: v up

    # ---- colour every texel of every used cell
    albedo = torch.zeros(size * size, 3, dtype=torch.uint8, device=dev)
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
    view_of = best_view[t_of]
    flat_idx = (py.long() * size + px.long())
    col = torch.zeros(len(pts), 3, device=dev)
    for k, v in enumerate(views):
        sel = ok & (view_of == k)
        if not bool(sel.any()):
            continue
        h, w = v.image.shape[:2]
        u, y, _ = _project(pts[sel], v)
        g = torch.stack([2 * u / (w - 1) - 1, 2 * y / (h - 1) - 1], -1)[None, None]
        img = v.image.permute(2, 0, 1)[None].float()
        col[sel] = F.grid_sample(img, g, mode="bilinear", align_corners=True, padding_mode="border")[0, :, 0].T
    unseen = ok & (view_of < 0)
    if fallback_rgb is not None and bool(unseen.any()):
        fb = torch.as_tensor(fallback_rgb, dtype=torch.float32, device=dev)
        if fb.max() <= 1.0:
            fb = fb * 255
        col[unseen] = (wa[unseen, None] * fb[ft[t_of[unseen], 0]] + wb[unseen, None] * fb[ft[t_of[unseen], 1]]
                       + wc[unseen, None] * fb[ft[t_of[unseen], 2]])  # fmt: skip
    albedo[flat_idx[ok]] = col[ok].round().clamp(0, 255).to(torch.uint8)
    info = {"triangles": int(n_f), "cell_px": int(cs), "views": len(views),
            "unseen_triangles": int((best_view < 0).sum())}  # fmt: skip
    return uv_obj.cpu().numpy(), albedo.view(size, size, 3).cpu().numpy(), info
