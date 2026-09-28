"""Merging per-pass models: the similarity fit, its composition, and the groups a set of links forms."""

from __future__ import annotations

import numpy as np

from drone3d.fastsfm.merge import compose, groups_from_links, invert, ransac_sim3, umeyama


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def test_similarity_is_recovered_through_outliers() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 3)) * 10
    s, r, t = 0.37, _rot(np.array([1.0, 2, 3]), 40), np.array([5.0, -2, 1])
    b = s * a @ r.T + t + rng.normal(scale=0.01, size=a.shape)
    b[:120] = rng.normal(size=(120, 3)) * 10  # 30 % wrong correspondences
    assert np.allclose(umeyama(a[120:], b[120:])[1], r, atol=1e-3)
    fit = ransac_sim3(a, b, thresh=0.1)
    fs, fr, ft = fit["sim3"]
    assert abs(fs - s) < 1e-3 and np.allclose(fr, r, atol=1e-3) and np.allclose(ft, t, atol=1e-2)
    assert 270 <= fit["inliers"] <= 285


def test_compose_and_invert() -> None:
    p = (2.0, _rot(np.array([0, 0, 1.0]), 30), np.array([1.0, 0, 0]))
    q = (0.5, _rot(np.array([1.0, 0, 0]), 20), np.array([0, 3.0, 0]))
    x = np.array([0.3, -1.2, 2.0])
    apply = lambda t, v: t[0] * t[1] @ v + t[2]  # noqa: E731
    assert np.allclose(apply(compose(p, q), x), apply(p, apply(q, x)))
    assert np.allclose(apply(compose(invert(p), p), x), x)


def test_groups_chain_transforms_to_the_largest_model() -> None:
    t_ab = (2.0, _rot(np.array([0, 1.0, 0]), 10), np.array([1.0, 0, 0]))  # b -> a
    t_bc = (0.5, _rot(np.array([1.0, 0, 0]), 25), np.array([0, 0, 2.0]))  # c -> b
    edges = [{"a": "a", "b": "b", "sim3": t_ab, "inliers": 900, "candidates": 4000},
             {"a": "b", "b": "c", "sim3": t_bc, "inliers": 800, "candidates": 4000},
             {"a": "a", "b": "d", "sim3": t_ab, "inliers": 120, "candidates": 4000}]  # too weak to merge
    groups = groups_from_links(["a", "b", "c", "d"], edges)
    assert [sorted(g["members"]) for g in groups] == [["a", "b", "c"], ["d"]]
    x = np.array([1.0, 2.0, 3.0])
    apply = lambda t, v: t[0] * t[1] @ v + t[2]  # noqa: E731
    assert np.allclose(apply(groups[0]["members"]["c"], x), apply(t_ab, apply(t_bc, x)))  # c -> b -> a
