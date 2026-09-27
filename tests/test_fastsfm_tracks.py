"""Flow tracks follow a known motion exactly and end where the direct check disagrees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone3d.fastsfm.tracks import build_tracks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class ExactFlow:
    """Frame ``i`` is filled with the value ``i``; the scene translates by ``v`` px per frame."""

    def __init__(self, v: tuple[float, float], broken: tuple[int, int] | None = None) -> None:
        self.v = torch.tensor(v)
        self.broken = broken  # a direct pair whose flow is off by 3 px

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n, h, w, _ = a.shape
        ia, ib = a[:, 0, 0, 0].long(), b[:, 0, 0, 0].long()
        d = (ib - ia).float()[:, None] * self.v.to(a.device)[None]
        if self.broken is not None:
            bad = (ia == self.broken[0]) & (ib == self.broken[1])
            d[bad] += 3.0
            bad = (ia == self.broken[1]) & (ib == self.broken[0])
            d[bad] -= 3.0
        return d[:, :, None, None].expand(n, 2, h, w).contiguous()


def _frames(n: int, h: int = 64, w: int = 96) -> torch.Tensor:
    return torch.arange(n, dtype=torch.uint8, device="cuda")[:, None, None, None].expand(n, h, w, 3).contiguous()


def test_tracks_follow_a_translation() -> None:
    v = (2.5, -0.75)
    tr = build_tracks(_frames(6), ExactFlow(v), span=3, stride=8)
    assert tr.stats["ended_by_direct_check"] == 0
    first = {}
    for img, tid, xy in zip(tr.image, tr.track, tr.xy, strict=True):
        if tid not in first:
            first[tid] = (img, xy)
        else:
            i0, xy0 = first[tid]
            np.testing.assert_allclose(xy, xy0 + (img - i0) * np.array(v), atol=1e-3)
    # every track is observed in consecutive images only
    for tid in np.unique(tr.track):
        imgs = np.sort(tr.image[tr.track == tid])
        assert np.all(np.diff(imgs) == 1)
    assert tr.stats["mean_track_length"] > 3


def test_direct_check_ends_drifting_tracks() -> None:
    # The direct flow 1 -> 3 disagrees with the chained 1 -> 2 -> 3 by 3 px:
    # tracks alive at image 1 must not reach image 3 -- unless the direct
    # prediction leaves the frame (x > 95 - 5), where it cannot be checked.
    tr = build_tracks(_frames(5), ExactFlow((1.0, 0.0), broken=(1, 3)), span=3, stride=8, max_dev_px=1.0)
    assert tr.stats["ended_by_direct_check"] > 0
    at1 = tr.image == 1
    checkable = set(tr.track[at1 & (tr.xy[:, 0] <= 90)])
    unchecked = set(tr.track[at1 & (tr.xy[:, 0] > 90)])
    at3 = set(tr.track[tr.image == 3])
    assert checkable and not (checkable & at3)
    assert unchecked <= at3 | set(tr.track[tr.image == 2])
