"""OpenCV edge-aware depth refinement: empty pixels stay empty, a clean plane stays a plane, bad names fail."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
from drone3d.fastsfm.refine import refine_depths  # noqa: E402


@pytest.mark.parametrize("method", ["guided", "fgs"])
def test_refinement_keeps_empty_pixels_empty_and_planes_flat(method: str) -> None:
    rng = np.random.default_rng(0)
    h, w = 90, 160
    depth = np.full((h, w), 20.0, np.float32) * (1 + rng.normal(0, 0.01, (h, w))).astype(np.float32)
    depth[:20] = 0.0  # sky
    rgb = np.full((h, w, 3), 120, np.uint8)
    tri = depth.copy()
    tri[:, ::2] = 0.0  # half the pixels triangulated, the rest filled
    (out,) = refine_depths([depth], rgb[None], [tri], method=method)
    assert (out[:20] == 0).all()
    assert np.abs(out[20:] - 20.0).max() < 1.0
    assert np.std(out[20:]) < np.std(depth[20:])  # smoother than its input


def test_unknown_method_and_none() -> None:
    d = [np.ones((8, 8), np.float32)]
    assert refine_depths(d, np.zeros((1, 8, 8, 3), np.uint8), d, method="none") is d
    with pytest.raises(ValueError):
        refine_depths(d, np.zeros((1, 8, 8, 3), np.uint8), d, method="bilateral-solver")
