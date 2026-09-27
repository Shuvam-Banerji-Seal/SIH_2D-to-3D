"""The GPU soup atlas colours each triangle from the view that sees it, at the right texel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone3d.export.texture_gpu import View, bake_soup_texture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _plane(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A 2 x 2 square at z = 5 split into 2 n^2 triangles."""
    xs = np.linspace(-1, 1, n + 1)
    gx, gy = np.meshgrid(xs, xs)
    v = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)], 1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return v, f


def test_texels_carry_the_image_colour_at_their_point() -> None:
    v, f = _plane()
    h, w = 200, 200
    # image: red grows with image x, green with image y -> the texture must reproduce that ramp
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), np.full_like(xx, 50)], -1).astype(np.uint8)
    view = View(fx=400.0, cx=100.0, cy=100.0, rotation=np.eye(3), translation=np.zeros(3),
                image=torch.as_tensor(img, device="cuda"))  # fmt: skip
    uv, albedo, info = bake_soup_texture(v, f, [view], size=512)
    assert info["unseen_triangles"] == 0 and uv.shape == (len(f), 3, 2)
    # sample each triangle's centroid in the atlas and compare with where the centroid projects
    cen_uv = uv.mean(1)
    tx = np.clip((cen_uv[:, 0] * 512).astype(int), 0, 511)
    ty = np.clip(((1 - cen_uv[:, 1]) * 512).astype(int), 0, 511)
    got = albedo[ty, tx].astype(float)
    cen3 = v[f].mean(1)
    u = 400 * cen3[:, 0] / cen3[:, 2] + 100 - 0.5
    y = 400 * cen3[:, 1] / cen3[:, 2] + 100 - 0.5
    want = np.stack([u * 255 / (w - 1), y * 255 / (h - 1)], 1)
    np.testing.assert_allclose(got[:, :2], want, atol=4.0)


def test_occluded_triangles_take_the_view_that_sees_them() -> None:
    v, f = _plane(2)
    front = np.c_[v[:, :2] * 0.5, np.full(len(v), 3.0)]  # a smaller plane in front of the first view
    verts = np.vstack([v, front])
    faces = np.vstack([f, f + len(v)])
    red = torch.zeros(100, 100, 3, dtype=torch.uint8, device="cuda")
    red[..., 0] = 255
    blue = torch.zeros(100, 100, 3, dtype=torch.uint8, device="cuda")
    blue[..., 2] = 255
    # view 0 looks straight at both planes (the back plane's centre is hidden); view 1 is off to the side
    rot1 = np.array([[np.cos(0.6), 0, -np.sin(0.6)], [0, 1, 0], [np.sin(0.6), 0, np.cos(0.6)]])
    views = [View(60.0, 50.0, 50.0, np.eye(3), np.zeros(3), red),
             View(60.0, 50.0, 50.0, rot1, -rot1 @ np.array([3.5, 0.0, 0.0]), blue)]  # fmt: skip
    uv, albedo, _ = bake_soup_texture(verts, faces, views, size=256, zbuf=64)
    cen_uv = uv.mean(1)
    tx = np.clip((cen_uv[:, 0] * 256).astype(int), 0, 255)
    ty = np.clip(((1 - cen_uv[:, 1]) * 256).astype(int), 0, 255)
    col = albedo[ty, tx]
    front_tris = slice(len(f), 2 * len(f))
    assert (col[front_tris, 0] > 200).all()  # the front plane faces view 0 squarely: red
