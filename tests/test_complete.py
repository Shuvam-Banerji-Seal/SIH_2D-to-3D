"""The complete model's geometry: flat facades, a closed solid, invented structure carved, unseen ground filled."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.complete import carve, fill_ground, snap_planes, solidify


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(xs, xs)
    v = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], 1)
    q = np.arange(n * n).reshape(n, n)
    a, b, c, d = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    return v, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def _box(lo, hi, n: int = 16, faces: tuple[str, ...] = ("-x", "+x", "-y", "+y", "-z", "+z")):  # type: ignore[no-untyped-def]
    """A box of fine grids, outward-facing, the listed faces only (vertices duplicated along the edges)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    vs, fs, off = [], [], 0
    for name in faces:
        axis, side = "xyz".index(name[1]), name[0] == "+"
        u, w = [k for k in range(3) if k != axis]
        g, f = _grid(n)
        p = np.zeros_like(g)
        p[:, u] = lo[u] + g[:, 0] * (hi[u] - lo[u])
        p[:, w] = lo[w] + g[:, 1] * (hi[w] - lo[w])
        p[:, axis] = hi[axis] if side else lo[axis]
        # outward: the grid's (u, w) normal is +axis when (u, w, axis) is a right-handed cycle
        flip = (side != ((u, w, axis) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))))
        vs.append(p)
        fs.append((f[:, ::-1] if flip else f) + off)
        off += len(p)
    return np.concatenate(vs), np.concatenate(fs)


def test_snap_planes_flattens_relief_and_keeps_the_corners() -> None:
    v, f = _box([0, 0, 0], [4, 4, 6], n=24)
    rng = np.random.default_rng(0)
    bumpy = v + rng.normal(0, 0.03, v.shape)  # window relief ~0.5 % of the size
    out, planes = snap_planes(bumpy, f)
    assert len(planes) >= 6
    for axis, value in ((0, 0.0), (0, 4.0), (1, 0.0), (1, 4.0), (2, 6.0)):
        on = np.abs(v[:, axis] - value) < 1e-9
        assert np.abs(out[on, axis] - value).max() < 0.03  # flat, where the noise was +-0.1
    assert np.linalg.norm(out - bumpy, axis=1).max() < 0.5  # nothing flung away


def test_snap_planes_leaves_a_sphere_alone() -> None:
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=4)
    out, planes = snap_planes(np.asarray(s.vertices), np.asarray(s.faces))
    assert planes == [] and np.allclose(out, s.vertices)


def test_solidify_closes_an_open_top_into_a_solid() -> None:
    v, f = _box([0, 0, 0], [2, 2, 3], faces=("-x", "+x", "-y", "+y", "-z"))  # carving opened the roof
    sv, sf, info = solidify(v, f, resolution=96)
    import trimesh

    m = trimesh.Trimesh(sv, sf, process=False)
    assert m.is_watertight and info["watertight"]
    assert info["fill"] == "enclosed on two axes"
    assert m.volume == pytest.approx(12.0, rel=0.1)  # a solid, not a shell
    from drone3d.complete import _closed_as_stored

    assert _closed_as_stored(sv, sf)  # closed as a file stores it, positions merged in float32
    flat, _ = snap_planes(sv, sf, tol=0.03, corners=False)
    assert _closed_as_stored(flat, sf)  # snapped along the normals only, it stays closed


def test_solidify_keeps_a_courtyard_open_when_the_walls_are_closed() -> None:
    """A closed ring of walls round a courtyard open to the sky: the flood fill works, the courtyard stays empty."""
    import trimesh

    walls = [([0, 0, 0], [6, 1.5, 2]), ([0, 4.5, 0], [6, 6, 2]), ([0, 1.5, 0], [1.5, 4.5, 2]), ([4.5, 1.5, 0], [6, 4.5, 2])]
    parts = [_box(lo, hi, n=12) for lo, hi in walls]
    v = np.concatenate([p[0] for p in parts])
    f = np.concatenate([p[1] + sum(len(q[0]) for q in parts[:i]) for i, p in enumerate(parts)])
    sv, sf, info = solidify(v, f, resolution=96)
    assert info["fill"] == "flood fill"
    assert trimesh.Trimesh(sv, sf).volume == pytest.approx(54.0, rel=0.12)  # 72 if the courtyard were filled


def _view(centre: np.ndarray, target: np.ndarray, size: tuple[int, int] = (320, 240), f: float = 300.0):  # type: ignore[no-untyped-def]
    import torch

    from drone3d.export.texture_gpu import View

    fwd = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    r = np.stack([right, np.cross(fwd, right), fwd])
    return View(f, size[0] / 2, size[1] / 2, r, -r @ centre, torch.zeros(size[1], size[0], 3, dtype=torch.uint8))


def test_carving_removes_what_the_camera_saw_through() -> None:
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    wall_v, wall_f = _box([0, 0, 0], [0.2, 4, 4], faces=("-x",))  # the measured facade, seen from -x
    slab_v, slab_f = _box([-1.0, 1.5, 2.0], [-0.9, 2.5, 2.1], n=4)  # an invented canopy in front of it
    back_v, back_f = _box([4, 0, 0], [4.2, 4, 4], n=6, faces=("+x",))  # a generated wall no camera sees
    gv = np.concatenate([slab_v, back_v])
    gf = np.concatenate([slab_f, back_f + len(slab_v)])
    views = [_view(np.array([-8.0, y, 2.2]), np.array([0.0, 2.0, 2.0])) for y in (1.5, 2.0, 2.5)]
    keep = carve(gv, gf, views, (wall_v, wall_f), device=dev)
    assert not keep[: len(slab_f)].any()  # the canopy: seen through by all three views
    assert keep[len(slab_f) :].all()  # the far wall: behind the facade, never seen


def test_ground_is_filled_where_nothing_covers_it() -> None:
    v, f = _grid(41)
    v = v * 20 - 10  # a 20 x 20 ground at z ~ -10 .. no: flat at 0
    v[:, 2] = 0.0
    cen = v[f].mean(1)
    hole = np.hypot(cen[:, 0] - 3, cen[:, 1]) < 3  # the far side, never seen
    f = f[~hole]
    colours = np.full((len(f), 3), 120, np.uint8)
    gv, gf, gc = fill_ground(v, f, colours, np.array([0.0, 0.0, 0.0]), 8.0, 0.25)
    assert len(gf)
    c = gv[gf].mean(1)
    assert (np.hypot(c[:, 0] - 3, c[:, 1]) < 3.5).mean() > 0.5  # the fill is where the hole was
    assert np.abs(gv[:, 2]).max() < 0.05 and (gc == 120).all()


def test_the_platform_a_generator_stood_the_object_on_goes() -> None:
    from drone3d.complete import drop_base_slab

    house_v, house_f = _box([4, 4, 0], [8, 8, 6], n=12)  # the object, standing from 0 to 6
    slab_v, slab_f = _box([0, 0, 1.8], [12, 12, 2.0], n=24)  # a lake slab at a third of its height
    v = np.concatenate([house_v, slab_v])
    f = np.concatenate([house_f, slab_f + len(house_v)])
    keep = drop_base_slab(v, f)
    cen = v[f].mean(1)
    assert not keep[len(house_f) :][~((abs(cen[len(house_f) :, 0] - 6) < 2.2) & (abs(cen[len(house_f) :, 1] - 6) < 2.2))].any()
    assert not keep[: len(house_f)][cen[: len(house_f), 2] < 1.7].any()  # under the water line: not the object
    assert keep[: len(house_f)][cen[: len(house_f), 2] > 2.2].all()  # what stands on it stays
    alone = drop_base_slab(house_v, house_f)
    assert alone.all()  # no platform, nothing goes (the roof is horizontal, but at the top)


def test_what_one_generation_alone_invents_is_voted_out() -> None:
    """Three generations of a tower; one adds a wing no other has: the consensus keeps the tower alone."""
    import trimesh

    from drone3d.complete import solidify_consensus

    tower = _box([0, 0, 0], [2, 2, 6], n=12)
    wing_v, wing_f = _box([2, 0, 0], [5, 2, 2], n=12)
    with_wing = (np.concatenate([tower[0], wing_v]), np.concatenate([tower[1], wing_f + len(tower[0])]))
    sv, sf, info = solidify_consensus([tower, tower, with_wing], resolution=96)
    m = trimesh.Trimesh(sv, sf, process=False)
    assert info["consensus"] == {"of": 3, "votes": 2} and m.is_watertight
    assert m.volume == pytest.approx(24.0, rel=0.1)  # the tower's 2 x 2 x 6; the wing (12 more) is gone
    union, _, _ = solidify_consensus([tower, tower, with_wing], resolution=96, votes=1)
    assert union[:, 0].max() > 4.5  # voted with one, the wing would stay
