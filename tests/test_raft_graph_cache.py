"""RaftFlow keeps a bounded number of captured CUDA graphs (each holds a private memory pool)."""

from __future__ import annotations

import pytest
import torch

from drone3d.keyframes.flow import RaftFlow

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


class _Net(torch.nn.Module):
    def forward(self, a, b, num_flow_updates=12):  # type: ignore[no-untyped-def]
        return [(a - b)[:, :2] * 0.5]


def test_least_recently_used_graph_is_dropped() -> None:
    f = RaftFlow("raft_small", batch=2, iters=1, max_graphs=2, net=_Net().cuda())
    frames = {hw: torch.randint(0, 255, (2, *hw, 3), dtype=torch.uint8, device="cuda") for hw in [(8, 16), (16, 16), (8, 8)]}
    f(frames[(8, 16)], frames[(8, 16)])
    f(frames[(16, 16)], frames[(16, 16)])
    f(frames[(8, 16)], frames[(8, 16)])  # used again: now the most recent
    out = f(frames[(8, 8)], torch.zeros_like(frames[(8, 8)]))
    assert list(f._graphs) == [(8, 16), (8, 8)]
    ref = (frames[(8, 8)].permute(0, 3, 1, 2).float() / 127.5 - 1.0 + 1.0)[:, :2] * 0.5
    assert torch.allclose(out, ref, atol=1e-2)
