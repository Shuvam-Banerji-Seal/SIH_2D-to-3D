"""Flow triangulation recovers exact depth for known cameras, and rejects what it cannot see."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone3d.fastsfm.dense import Camera, _project, fuse_depths, pair_depth

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _yaw(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _exact_flow(ci: Camera, cj: Camera, depth_of, h: int, w: int) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Flow i -> j for a scene whose camera-i depth at pixel (x, y) is ``depth_of(x, y)``."""
    from drone3d.fastsfm.dense import _rays

    py, px = torch.meshgrid(torch.arange(h, dtype=torch.float64, device="cuda"), torch.arange(w, dtype=torch.float64, device="cuda"), indexing="ij")
    rays = _rays(ci, px, py)
    pts = torch.as_tensor(ci.centre, device="cuda") + depth_of(px, py)[..., None] * rays
    u, v, _ = _project(cj, pts)
    return torch.stack([u - px, v - py]).float()


@pytest.mark.parametrize("k1", [0.0, 0.05])
def test_pair_depth_recovers_a_slanted_plane(k1: float) -> None:
    h, w = 48, 64
    ci = Camera(60.0, 32.0, 24.0, k1, np.eye(3), np.zeros(3))
    rot = _yaw(-4.0)
    cj = Camera(60.0, 32.0, 24.0, k1, rot, -rot @ np.array([1.0, 0.0, 0.0]))

    def depth_of(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        return 10.0 + 0.05 * px + 0.02 * py

    flow = _exact_flow(ci, cj, depth_of, h, w)
    depth, angle = pair_depth(ci, cj, flow, torch.ones(h, w, dtype=torch.bool, device="cuda"), min_angle_deg=1.0)
    truth = depth_of(*torch.meshgrid(torch.arange(w, device="cuda").double(), torch.arange(h, device="cuda").double(), indexing="xy")).float()
    ok = depth > 0
    assert ok.float().mean() > 0.95
    torch.testing.assert_close(depth[ok], truth[ok], rtol=2e-4, atol=0)
    assert float(angle[ok].min()) > 1.0


def test_pair_depth_rejects_pure_rotation_and_bad_flow() -> None:
    h, w = 32, 48
    ci = Camera(50.0, 24.0, 16.0, 0.0, np.eye(3), np.zeros(3))
    rot = _yaw(5.0)
    cj = Camera(50.0, 24.0, 16.0, 0.0, rot, np.zeros(3))  # same centre: no parallax
    flow = _exact_flow(ci, cj, lambda px, py: torch.full_like(px, 10.0), h, w)
    depth, _ = pair_depth(ci, cj, flow, torch.ones(h, w, dtype=torch.bool, device="cuda"))
    assert not bool((depth > 0).any())
    # with a baseline, a flow off the epipolar line by 3 px is rejected
    cj2 = Camera(50.0, 24.0, 16.0, 0.0, np.eye(3), np.array([-1.0, 0.0, 0.0]))
    flow2 = _exact_flow(ci, cj2, lambda px, py: torch.full_like(px, 10.0), h, w)
    flow2[1] += 3.0
    depth2, _ = pair_depth(ci, cj2, flow2, torch.ones(h, w, dtype=torch.bool, device="cuda"), max_reproj_px=1.5)
    assert not bool((depth2 > 0).any())


def test_fuse_depths_needs_agreeing_views() -> None:
    d = torch.tensor([[[10.0, 10.0]], [[10.2, 0.0]], [[30.0, 0.0]]], device="cuda")  # [K=3, H=1, W=2]
    wts = torch.ones_like(d)
    out = fuse_depths(d, wts, rel_tol=0.03, min_views=2)
    assert abs(float(out[0, 0]) - 10.1) < 1e-4  # the 30 m outlier is ignored
    assert float(out[0, 1]) == 0.0  # one view only


def test_extraction_weight_asks_for_min_views() -> None:
    from drone3d.fastsfm.dense_stage import extraction_weight

    assert extraction_weight("auto", 5) == 1.5 and extraction_weight("auto", 12) == 1.5  # two views
    assert extraction_weight("auto", 13) == 2.5 and extraction_weight("auto", 200) == 2.5  # three
    assert extraction_weight("4", 5) == 3.5 and extraction_weight(1, 40) == 0.5


def test_tie_depths_rasterises_tie_points_at_their_camera_depth() -> None:
    from types import SimpleNamespace as NS

    from drone3d.fastsfm.dense_stage import tie_depths

    pose = NS(rotation=NS(matrix=lambda: np.eye(3)), translation=np.array([0.0, 0.0, 2.0]))
    pts = {1: NS(xyz=np.array([0.0, 0.0, 3.0])), 2: NS(xyz=np.array([1.0, 0.0, 8.0])), 3: NS(xyz=np.array([0.0, 0.0, -5.0]))}
    obs = [NS(xy=np.array([20.5, 10.5]), point3D_id=1, has_point3D=lambda: True),
           NS(xy=np.array([61.0, 30.9]), point3D_id=2, has_point3D=lambda: True),
           NS(xy=np.array([5.0, 5.0]), point3D_id=3, has_point3D=lambda: True),   # behind the camera
           NS(xy=np.array([7.0, 7.0]), point3D_id=-1, has_point3D=lambda: False),  # not triangulated
           NS(xy=np.array([900.0, 5.0]), point3D_id=1, has_point3D=lambda: True)]  # outside the image
    im = NS(points2D=obs, cam_from_world=lambda: pose)
    (d,) = tie_depths(NS(points3D=pts), [im], 0.5, (40, 20))
    assert d.shape == (20, 40) and (d > 0).sum() == 2
    assert d[5, 10] == 5.0 and d[15, 30] == 10.0  # x, y halved; depth = z + t_z


def test_specks_are_removed_but_the_surface_stays() -> None:
    import open3d as o3d

    from drone3d.fastsfm.dense_stage import _drop_specks

    ground = o3d.geometry.TriangleMesh.create_box(10, 10, 0.1).subdivide_midpoint(4)  # the surface
    speck = o3d.geometry.TriangleMesh.create_tetrahedron(0.05).translate((5, 5, 2))  # a floater
    mesh = ground + speck
    before = len(mesh.triangles)
    out = _drop_specks(mesh, share=0.001, least=10)
    assert out == {"components": 1, "triangles": 4} and len(mesh.triangles) == before - 4
