"""The clean model: the ground comes out a flat terrain, a building on it stays and is smoothed, floaters go."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.export.terrain import _bilateral_numpy, clean_model, terrain_surface


def _grid(n: int, size: float, z: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A regular triangulated square [0, size]^2 of n x n vertices."""
    xs = np.linspace(0, size, n)
    xx, yy = np.meshgrid(xs, xs)
    v = np.stack([xx.ravel(), yy.ravel(), np.broadcast_to(z, xx.shape).ravel()], 1)
    q = np.arange(n * n).reshape(n, n)
    a, b, c, d = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    return v, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def _box(lo: np.ndarray, hi: np.ndarray, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A closed box with each face a fine n x n grid (a fused wall has many small facets)."""
    vs, fs, off = [], [], 0
    for axis in range(3):
        for side in (0, 1):
            u, w = [k for k in range(3) if k != axis]
            g, f = _grid(n, 1.0)
            p = np.zeros_like(g)
            p[:, u] = lo[u] + g[:, 0] * (hi[u] - lo[u])
            p[:, w] = lo[w] + g[:, 1] * (hi[w] - lo[w])
            p[:, axis] = hi[axis] if side else lo[axis]
            vs.append(p)
            fs.append(f + off)
            off += len(p)
    return np.concatenate(vs), np.concatenate(fs)


def _scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noisy ground 40 x 40, a 6 x 6 x 8 building in the middle, a small floater above one corner."""
    rng = np.random.default_rng(seed)
    gv, gf = _grid(81, 40.0)
    gv[:, 2] += rng.normal(0, 0.05, len(gv))
    bv, bf = _box(np.array([17.0, 17.0, 0.0]), np.array([23.0, 23.0, 8.0]))
    bv += rng.normal(0, 0.08, bv.shape)
    fv, ff = _box(np.array([3.0, 3.0, 6.0]), np.array([3.3, 3.3, 6.3]), n=2)
    v = np.concatenate([gv, bv, fv])
    f = np.concatenate([gf, bf + len(gv), ff + len(gv) + len(bv)])
    return v, f


def test_the_opening_removes_a_building_from_the_ground() -> None:
    xs = np.linspace(0, 40, 161)
    xx, yy = np.meshgrid(xs, xs)
    z = np.where((abs(xx - 20) < 3) & (abs(yy - 20) < 3), 8.0, 0.0)
    ground, data, (x0, y0) = terrain_surface(np.stack([xx.ravel(), yy.ravel(), z.ravel()], 1), 0.5, 10.0)
    assert data.all()
    assert abs(ground).max() < 0.2  # the roof is not ground


def test_clean_model_keeps_the_building_flattens_the_ground_and_drops_the_floater() -> None:
    v, f = _scene()
    cv, cf, info = clean_model(v, f, voxel=0.25, window=12.0)
    assert info["object_triangles"] > 0 and info["terrain_triangles"] > 0
    assert info["object_fragments_dropped"] >= 1
    assert info["terrain_triangles"] + info["object_triangles"] == len(cf)
    terrain = cv[np.unique(cf[: info["terrain_triangles"]])]
    assert abs(terrain[:, 2]).max() < 0.1  # the noise was +-0.15
    obj = cv[np.unique(cf[info["terrain_triangles"] :])]
    assert obj[:, 2].max() > 7.5  # the building stands
    assert not ((obj[:, 0] < 5) & (obj[:, 1] < 5)).any()  # the floater is gone
    # the building's east wall (x = 23) is flatter than it was
    raw_wall = v[(abs(v[:, 0] - 23) < 0.3) & (v[:, 2] > 1) & (v[:, 2] < 7) & (abs(v[:, 1] - 20) < 2.5)]
    new_wall = obj[(abs(obj[:, 0] - 23) < 0.3) & (obj[:, 2] > 1) & (obj[:, 2] < 7) & (abs(obj[:, 1] - 20) < 2.5)]
    assert new_wall[:, 0].std() < 0.6 * raw_wall[:, 0].std()


def test_terrain_faces_point_up() -> None:
    v, f = _scene()
    cv, cf, info = clean_model(v, f, voxel=0.25, window=12.0)
    t = cv[cf[: info["terrain_triangles"]]]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    assert (n[:, 2] > 0).all()


def test_bilateral_smoothing_on_the_gpu_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from drone3d.export.terrain import _bilateral_torch

    v, f = _box(np.array([0.0, 0.0, 0.0]), np.array([4.0, 4.0, 4.0]), n=8)
    v = v + np.random.default_rng(1).normal(0, 0.05, v.shape)
    kw = {"normal_iters": 6, "vertex_iters": 10, "sigma_r": 0.35}
    assert np.allclose(_bilateral_torch(v, f, **kw), _bilateral_numpy(v, f, **kw), atol=1e-6)
