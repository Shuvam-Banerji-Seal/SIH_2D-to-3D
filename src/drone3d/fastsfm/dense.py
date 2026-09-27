"""Dense depth from optical flow and known poses, fused into a mesh and a point cloud.

Once SfM has the poses, every consistent flow vector between two keyframes is
a correspondence to triangulate. Per reference keyframe and neighbour this
gives a depth map directly, gated by the triangulation angle and the
reprojection error in both views; depths from several neighbours are fused
by a weighted median. Open3D's GPU TSDF then integrates all depth maps into a
coloured surface.

Coordinates: flow and depth maps index pixel centres at integers; COLMAP
intrinsics put the top-left pixel centre at (0.5, 0.5), hence the ``+0.5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Camera", "fuse_depths", "pair_depth", "tsdf_fuse"]


@dataclass
class Camera:
    """Pinhole + one radial term (COLMAP SIMPLE_RADIAL / RADIAL k1), world-to-camera pose."""

    f: float
    cx: float
    cy: float
    k1: float
    rotation: np.ndarray  # [3, 3] cam_from_world
    translation: np.ndarray  # [3]

    @property
    def centre(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    @classmethod
    def from_colmap(cls, image, camera) -> Camera:  # type: ignore[no-untyped-def]
        p = list(camera.params)
        k1 = p[3] if len(p) > 3 else 0.0
        pose = image.cam_from_world()
        return cls(p[0], p[1], p[2], k1, np.asarray(pose.rotation.matrix()), np.asarray(pose.translation))


def _undistort(xd: torch.Tensor, yd: torch.Tensor, k1: float, iters: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = xd.clone(), yd.clone()
    for _ in range(iters):
        s = 1 + k1 * (x * x + y * y)
        x, y = xd / s, yd / s
    return x, y


def _rays(cam: Camera, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """World-frame ray directions ``[..., 3]`` with camera-frame z = 1 for pixel coords (centres at integers)."""
    xn, yn = _undistort((px + 0.5 - cam.cx) / cam.f, (py + 0.5 - cam.cy) / cam.f, cam.k1)
    d_cam = torch.stack([xn, yn, torch.ones_like(xn)], -1)
    rot = torch.as_tensor(cam.rotation, dtype=px.dtype, device=px.device)
    return d_cam @ rot  # R^T d, row-vector form


def _project(cam: Camera, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot = torch.as_tensor(cam.rotation, dtype=pts.dtype, device=pts.device)
    t = torch.as_tensor(cam.translation, dtype=pts.dtype, device=pts.device)
    pc = pts @ rot.T + t
    z = pc[..., 2]
    x, y = pc[..., 0] / z, pc[..., 1] / z
    s = 1 + cam.k1 * (x * x + y * y)
    return cam.f * x * s + cam.cx - 0.5, cam.f * y * s + cam.cy - 0.5, z


@torch.inference_mode()
def pair_depth(
    cam_i: Camera,
    cam_j: Camera,
    flow_ij: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_angle_deg: float = 1.0,
    max_reproj_px: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth of every pixel of image ``i`` triangulated with its flow into ``j``.

    Args:
        flow_ij: ``[2, H, W]`` flow from ``i`` to ``j``.
        valid: ``[H, W]`` bool, forward-backward consistent pixels.
    Returns:
        ``(depth [H, W], angle_deg [H, W])``; depth is 0 where rejected.
    """
    _, h, w = flow_ij.shape
    dev = flow_ij.device
    py, px = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float64), torch.arange(w, device=dev, dtype=torch.float64), indexing="ij")
    fx = flow_ij.double()
    qx, qy = px + fx[0], py + fx[1]
    di, dj = _rays(cam_i, px, py), _rays(cam_j, qx, qy)
    ci = torch.as_tensor(cam_i.centre, dtype=torch.float64, device=dev)
    cj = torch.as_tensor(cam_j.centre, dtype=torch.float64, device=dev)
    wv = cj - ci
    a11, a12, a22 = (di * di).sum(-1), (di * dj).sum(-1), (dj * dj).sum(-1)
    b1, b2 = (di * wv).sum(-1), (dj * wv).sum(-1)
    det = a11 * a22 - a12 * a12
    lam_i = (a22 * b1 - a12 * b2) / det
    lam_j = (a12 * b1 - a11 * b2) / det
    x = 0.5 * ((ci + lam_i[..., None] * di) + (cj + lam_j[..., None] * dj))
    ri, rj = x - ci, x - cj
    cosang = (ri * rj).sum(-1) / (ri.norm(dim=-1) * rj.norm(dim=-1)).clamp_min(1e-12)
    angle = torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))
    ui, vi, zi = _project(cam_i, x)
    uj, vj, zj = _project(cam_j, x)
    err = torch.maximum(torch.hypot(ui - px, vi - py), torch.hypot(uj - qx, vj - qy))
    ok = valid & (lam_i > 0) & (lam_j > 0) & (zi > 0) & (zj > 0) & (angle >= min_angle_deg) & (err <= max_reproj_px)
    depth = torch.where(ok, zi, torch.zeros_like(zi))
    return depth.float(), torch.where(ok, angle, torch.zeros_like(angle)).float()


@torch.inference_mode()
def fuse_depths(depths: torch.Tensor, weights: torch.Tensor, *, rel_tol: float = 0.03, min_views: int = 2) -> torch.Tensor:
    """Fuse ``[K, H, W]`` candidate depths (0 = none): median, then weighted mean of the agreeing ones.

    A pixel needs ``min_views`` candidates within ``rel_tol`` of the median, so
    a single wrong flow vector cannot place a surface on its own.
    """
    has = depths > 0
    d = torch.where(has, depths, torch.full_like(depths, float("nan")))
    med = torch.nanmedian(d, dim=0).values
    agree = has & ((depths - med[None]).abs() <= rel_tol * med[None])
    wsum = (weights * agree).sum(0)
    fused = (depths * weights * agree).sum(0) / wsum.clamp_min(1e-9)
    return torch.where((agree.sum(0) >= min_views) & (wsum > 0), fused, torch.zeros_like(fused))


def tsdf_fuse(
    frames: list[tuple[np.ndarray, np.ndarray, Camera]],
    *,
    voxel: float,
    depth_max: float,
    trunc_voxels: float = 4.0,
    block_count: int = 200_000,
):  # type: ignore[no-untyped-def]
    """Integrate ``(depth [H, W] float32, rgb [H, W, 3] uint8, camera)`` into a GPU TSDF.

    Returns Open3D's ``VoxelBlockGrid``; ``extract_triangle_mesh()`` and
    ``extract_point_cloud()`` give the outputs. Radial distortion is ignored
    here (depth maps are dense per pixel; the error is below a voxel at the
    working resolution for |k1| < 0.1).
    """
    import open3d as o3d
    import open3d.core as o3c

    device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel,
        block_resolution=16,
        block_count=block_count,
        device=device,
    )
    for depth, rgb, cam in frames:
        intr = o3c.Tensor(np.array([[cam.f, 0, cam.cx - 0.5], [0, cam.f, cam.cy - 0.5], [0, 0, 1]]), o3c.float64)
        extr = np.eye(4)
        extr[:3, :3], extr[:3, 3] = cam.rotation, cam.translation
        extr_t = o3c.Tensor(extr, o3c.float64)
        d_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth, dtype=np.float32))).to(device)
        c_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb.astype(np.float32) / 255.0))).to(device)
        blocks = vbg.compute_unique_block_coordinates(d_img, intr, extr_t, 1.0, depth_max, trunc_voxels)
        vbg.integrate(blocks, d_img, c_img, intr, intr, extr_t, 1.0, depth_max, trunc_voxels)
    return vbg
