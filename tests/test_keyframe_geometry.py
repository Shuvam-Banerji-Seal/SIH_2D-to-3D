"""Two-view geometry must tell 3D parallax from rotation / planar motion.

Synthetic scenes with a known answer: a camera that only rotates, a camera
translating over a plane (both explained by a homography), and a camera
translating over a scene with real depth (only an F explains it).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from drone3d.keyframes.geometry import (  # noqa: E402
    fit_homography,
    sampson_sq,
    transfer_sq,
    two_view_analysis,
)

FOCAL = 500.0
W, H = 640, 360


def _rot(axis: tuple[float, float, float], deg: float) -> torch.Tensor:
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    t = math.radians(deg)
    k = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def _project(points: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    cam = points @ r.T + t
    return torch.stack(
        [FOCAL * cam[:, 0] / cam[:, 2] + W / 2, FOCAL * cam[:, 1] / cam[:, 2] + H / 2], dim=-1
    )


def _scene(n: int, depth: tuple[float, float], planar: bool, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g, dtype=torch.float64) * W
    v = torch.rand(n, generator=g, dtype=torch.float64) * H
    if planar:
        z = torch.full((n,), sum(depth) / 2, dtype=torch.float64)
    else:
        z = depth[0] + torch.rand(n, generator=g, dtype=torch.float64) * (depth[1] - depth[0])
    return torch.stack([(u - W / 2) * z / FOCAL, (v - H / 2) * z / FOCAL, z], dim=-1)


def _pair(r: torch.Tensor, t: torch.Tensor, planar: bool, noise: float = 0.3, seed: int = 0):
    pts = _scene(1500, (20.0, 80.0), planar, seed)
    x = _project(pts, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    y = _project(pts, r, t)
    g = torch.Generator().manual_seed(seed + 1)
    x = x + noise * torch.randn(x.shape, generator=g, dtype=torch.float64)
    y = y + noise * torch.randn(y.shape, generator=g, dtype=torch.float64)
    return x.float(), y.float()


def _analyse(pairs):
    xs = torch.stack([p[0] for p in pairs])
    ys = torch.stack([p[1] for p in pairs])
    w = torch.ones(xs.shape[:2])
    return two_view_analysis(xs, ys, w, focal_px=FOCAL)


def test_rotation_only_and_plane_prefer_homography_parallax_prefers_f() -> None:
    rotation = _pair(_rot((0.2, 1.0, 0.1), 6.0), torch.zeros(3, dtype=torch.float64), planar=False)
    plane = _pair(_rot((0, 1, 0), 2.0), torch.tensor([3.0, 0.5, 0.0], dtype=torch.float64), True)
    parallax = _pair(
        _rot((0, 1, 0), 2.0), torch.tensor([3.0, 0.5, 0.0], dtype=torch.float64), False
    )
    fit = _analyse([rotation, plane, parallax])

    assert fit.prefers_3d.tolist() == [False, False, True]
    # Rotation and plane leave only the injected noise (~0.3 px per axis).
    assert fit.parallax_px[0] < 1.0
    assert fit.parallax_px[1] < 1.0
    # Real depth range 20..80 m under 3 m of baseline leaves tens of pixels.
    assert fit.parallax_px[2] > 5.0
    assert fit.parallax_deg[2] > 0.5


def test_homography_is_recovered_despite_outliers() -> None:
    x, y = _pair(_rot((0.3, 1.0, 0.0), 8.0), torch.zeros(3, dtype=torch.float64), False, 0.2)
    g = torch.Generator().manual_seed(7)
    bad = torch.rand(len(x), generator=g) < 0.25  # 25 % gross outliers
    y = y.clone()
    y[bad] += torch.randn(int(bad.sum()), 2, generator=g) * 40.0
    h = fit_homography(x[None], y[None], torch.ones(1, len(x)), sigma=0.5)
    err = transfer_sq(h, x[None], y[None])[0].sqrt()
    assert torch.median(err[~bad]) < 0.5


def test_fundamental_satisfies_epipolar_constraint_on_inliers() -> None:
    x, y = _pair(_rot((0, 1, 0), 3.0), torch.tensor([4.0, 1.0, 0.5], dtype=torch.float64), False)
    fit = _analyse([(x, y)])
    e = sampson_sq(fit.fundamental, x[None], y[None])[0].sqrt()
    assert torch.median(e) < 0.6
    assert fit.epipolar_rms_px[0] < 0.6


def test_padding_weights_are_ignored() -> None:
    x, y = _pair(_rot((0, 1, 0), 3.0), torch.tensor([4.0, 1.0, 0.5], dtype=torch.float64), False)
    garbage = torch.rand(500, 2) * 640
    xs = torch.cat([x, garbage])[None]
    ys = torch.cat([y, garbage.flip(0)])[None]
    w = torch.cat([torch.ones(len(x)), torch.zeros(500)])[None]
    padded = two_view_analysis(xs, ys, w, focal_px=FOCAL)
    clean = _analyse([(x, y)])
    assert padded.num_valid.item() == len(x)
    assert padded.prefers_3d.item() and clean.prefers_3d.item()
    assert abs(padded.parallax_px.item() - clean.parallax_px.item()) < 1e-3
