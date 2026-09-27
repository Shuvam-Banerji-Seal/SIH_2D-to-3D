"""The ray-cast renderer shows a textured quad where the camera sees it, and background elsewhere."""

from __future__ import annotations

import numpy as np

from drone3d.export.render_mesh import MeshRenderer


def test_quad_texture_and_background() -> None:
    # A 2 x 2 square at z = 5 facing the camera, textured with a red->green ramp along u.
    v = np.array([[-1, -1, 5], [1, -1, 5], [1, 1, 5], [-1, 1, 5]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    corner_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)[f]  # OBJ convention (v up)
    size = 64
    ramp = np.linspace(0, 255, size).astype(np.uint8)
    albedo = np.zeros((size, size, 3), np.uint8)
    albedo[..., 0] = ramp[None, :]  # red grows with u (atlas x)
    albedo[..., 1] = 255 - ramp[None, :]
    r = MeshRenderer(v, f, corner_uv, albedo, background=(0, 0, 255))
    img = r.render(40.0, 40.0, 40.0, np.eye(3), np.zeros(3), (80, 80))
    # the quad spans +-1 at depth 5 -> +-8 px around the centre (f = 40)
    centre = img[40, 40]
    assert centre[2] < 50 and abs(int(centre[0]) - 128) < 20  # on the quad, mid ramp
    left, right = img[40, 34], img[40, 46]
    assert right[0] > left[0] + 40  # red increases to the right, as u does
    assert tuple(img[5, 5]) == (0, 0, 255)  # background where no ray hits
