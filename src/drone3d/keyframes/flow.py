"""Dense optical flow (RAFT) between consecutive analysis frames, on the GPU.

Only consecutive pairs are ever computed -- forward ``t -> t+1`` and backward
``t+1 -> t`` in one batch -- and longer-range correspondence comes from
chaining them (:mod:`drone3d.keyframes.tracks`). That makes the cost linear in
the number of frames regardless of how far apart keyframes end up.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from drone3d.logging_utils import get_logger

__all__ = ["ConsecutiveFlow", "RaftFlow", "consistency_mask", "frame_sharpness"]

log = get_logger(__name__)


def _default_torch_home() -> None:
    # Weights live beside the other model files, not in ~/.cache.
    os.environ.setdefault("TORCH_HOME", "/store/huggingface/torch")


def _patch_raft(net: torch.nn.Module) -> None:
    """Make torchvision's RAFT inference launch-lean and CUDA-Graph capturable.

    Stock ``forward`` builds its coordinate grids on the CPU and copies them to
    the GPU, rebuilds the correlation lookup offsets on the CPU in each of the
    12 refinement iterations (a synchronous copy every time), divides by a CPU
    tensor, and upsamples the flow after every iteration although inference
    keeps only the last. Same arithmetic, minus those.
    """
    import types

    from torchvision.models.optical_flow import raft as raft_mod

    corr = net.corr_block
    side = 2 * corr.radius + 1

    def index_pyramid(self, centroids_coords):  # type: ignore[no-untyped-def]
        delta = getattr(self, "_ss_delta", None)
        if (
            delta is None
            or delta.device != centroids_coords.device
            or delta.dtype != centroids_coords.dtype
        ):
            r = torch.linspace(
                -self.radius,
                self.radius,
                side,
                device=centroids_coords.device,
                dtype=centroids_coords.dtype,
            )
            delta = torch.stack(torch.meshgrid(r, r, indexing="ij"), dim=-1).view(1, side, side, 2)
            self._ss_delta = delta
        b, _, h, w = centroids_coords.shape
        coords = centroids_coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
        levels = []
        for volume in self.corr_pyramid:
            levels.append(
                raft_mod.grid_sample(
                    volume, coords + delta, align_corners=True, mode="bilinear"
                ).view(b, h, w, -1)
            )
            coords = coords / 2
        return torch.cat(levels, dim=-1).permute(0, 3, 1, 2).contiguous()

    def compute_corr_volume(self, fmap1, fmap2):  # type: ignore[no-untyped-def]
        b, c, h, w = fmap1.shape
        out = torch.matmul(fmap1.view(b, c, h * w).transpose(1, 2), fmap2.view(b, c, h * w))
        return out.view(b, h, w, 1, h, w) / math.sqrt(c)

    def forward(self, image1, image2, num_flow_updates: int = 12):  # type: ignore[no-untyped-def]
        b, _, h, w = image1.shape
        fmap1, fmap2 = torch.chunk(
            self.feature_encoder(torch.cat([image1, image2], dim=0)), 2, dim=0
        )
        self.corr_block.build_pyramid(fmap1, fmap2)
        context_out = self.context_encoder(image1)
        hs = self.update_block.hidden_state_size
        hidden, context = torch.split(context_out, [hs, context_out.shape[1] - hs], dim=1)
        hidden, context = torch.tanh(hidden), F.relu(context)
        coords0 = raft_mod.make_coords_grid(
            b, h // 8, w // 8, device=fmap1.device
        )  # fp32, as stock
        coords1 = coords0.clone()
        for _ in range(num_flow_updates):
            feats = self.corr_block.index_pyramid(centroids_coords=coords1)
            hidden, delta_flow = self.update_block(hidden, context, feats, coords1 - coords0)
            coords1 = coords1 + delta_flow
        up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden)
        return [raft_mod.upsample_flow(flow=coords1 - coords0, up_mask=up_mask)]

    corr.index_pyramid = types.MethodType(index_pyramid, corr)
    corr._compute_corr_volume = types.MethodType(compute_corr_volume, corr)
    net.forward = types.MethodType(forward, net)


class RaftFlow:
    """Batched RAFT inference in bfloat16, replayed from a CUDA Graph.

    Kernel launches, not arithmetic, bound stock RAFT at this resolution: a
    call costs about the same whatever the batch. The network is therefore
    captured once per input shape and replayed with a single launch; the last
    chunk of a call is padded to the captured batch size.

    Args:
        model: ``raft_large`` (accurate) or ``raft_small`` (about 2x faster).
        batch: pairs per replay.
        iters: RAFT refinement iterations.
        cuda_graph: capture and replay (falls back to eager if capture fails).
    """

    def __init__(
        self,
        model: str = "raft_large",
        *,
        batch: int = 32,
        iters: int = 12,
        device: str | torch.device = "cuda",
        cuda_graph: bool = True,
    ) -> None:
        _default_torch_home()
        from torchvision.models import optical_flow as of

        if model == "raft_large":
            net = of.raft_large(weights=of.Raft_Large_Weights.DEFAULT)
        elif model == "raft_small":
            net = of.raft_small(weights=of.Raft_Small_Weights.DEFAULT)
        else:
            raise ValueError(f"unknown flow model {model!r} (raft_large | raft_small)")
        _patch_raft(net)
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.batch = max(1, batch)
        self.iters = iters
        self.name = model
        self.use_graph = cuda_graph and self.device.type == "cuda"
        self._graphs: dict[
            tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    @staticmethod
    def _prep(frames: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, H, W, 3]`` -> float ``[B, 3, H, W]`` in [-1, 1]."""
        return frames.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).contiguous()

    def _eager(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return self.net(a, b, num_flow_updates=self.iters)[-1].float()

    def _graph(self, h: int, w: int):  # type: ignore[no-untyped-def]
        key = (h, w)
        if key not in self._graphs:
            sa = torch.zeros(self.batch, 3, h, w, device=self.device)
            sb = torch.zeros_like(sa)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):  # warm up cuDNN autotuning and allocator outside capture
                    self._eager(sa, sb)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._eager(sa, sb)
            self._graphs[key] = (graph, sa, sb, out)
        return self._graphs[key]

    @torch.inference_mode()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Flow from ``a`` to ``b`` (uint8 ``[B, H, W, 3]`` on device) -> ``[B, 2, H, W]`` fp32."""
        n, h, w, _ = a.shape
        out = torch.empty(n, 2, h, w, device=self.device)
        if self.use_graph:
            try:
                graph, sa, sb, res = self._graph(h, w)
            except RuntimeError as exc:
                log.warning("CUDA Graph capture failed (%s); running RAFT eagerly", str(exc)[:120])
                self.use_graph = False
        for s in range(0, n, self.batch):
            e = min(n, s + self.batch)
            if self.use_graph:
                sa[: e - s].copy_(self._prep(a[s:e]))
                sb[: e - s].copy_(self._prep(b[s:e]))
                graph.replay()
                out[s:e] = res[: e - s]
            else:
                out[s:e] = self._eager(self._prep(a[s:e]), self._prep(b[s:e]))
        return out


def consistency_mask(fwd: torch.Tensor, bwd: torch.Tensor) -> torch.Tensor:
    """Forward-backward check (Sundaram et al., ECCV 2010), ``[B, H, W]`` bool.

    A pixel is consistent when following ``fwd`` then ``bwd`` returns within
    ``0.01 (|f|^2 + |b|^2) + 0.5`` px^2 of where it started and it lands inside
    the next frame.
    """
    b, _, h, w = fwd.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=fwd.device, dtype=fwd.dtype),
        torch.arange(w, device=fwd.device, dtype=fwd.dtype),
        indexing="ij",
    )
    tx = xs[None] + fwd[:, 0]
    ty = ys[None] + fwd[:, 1]
    grid = torch.stack([2 * tx / (w - 1) - 1, 2 * ty / (h - 1) - 1], dim=-1)
    bwd_at = F.grid_sample(bwd, grid, mode="bilinear", align_corners=True, padding_mode="border")
    diff = (fwd + bwd_at).pow(2).sum(1)
    bound = 0.01 * (fwd.pow(2).sum(1) + bwd_at.pow(2).sum(1)) + 0.5
    inside = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
    return (diff < bound) & inside


def frame_sharpness(frames: torch.Tensor) -> torch.Tensor:
    """Variance of the Laplacian of luma per frame (uint8 ``[B, H, W, 3]``) -> ``[B]``."""
    rgb = frames.float()
    luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[:, None]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=luma.dtype, device=luma.device)
    lap = F.conv2d(luma, kernel[None, None])
    return lap.flatten(1).var(dim=1)


@dataclass
class ConsecutiveFlow:
    """Flow between every consecutive pair of analysis frames, kept on the GPU.

    ``fwd[i]`` maps frame ``i`` to ``i + 1`` and ``bwd[i]`` maps ``i + 1`` to
    ``i`` (fp16, ``[N-1, 2, H, W]``). ``consistency[i]`` is the fraction of
    frame ``i`` that passes the forward-backward check -- near 1 inside a shot,
    near 0 across a cut. ``sharpness``, ``luma_mean`` and ``luma_std`` are per
    frame.
    """

    fwd: torch.Tensor
    bwd: torch.Tensor
    consistency: np.ndarray
    sharpness: np.ndarray
    luma_mean: np.ndarray
    luma_std: np.ndarray
    seconds: float
    frames: torch.Tensor | None = None  # uint8 [N, H, W, 3] analysis frames (GPU), for direct flow

    @property
    def num_frames(self) -> int:
        return len(self.sharpness)

    @classmethod
    def compute(
        cls,
        chunks: Iterable[tuple[np.ndarray, torch.Tensor]],
        flow: RaftFlow,
        *,
        progress: Callable[[int], None] | None = None,
        keep_frames: bool = True,
        capacity: int | None = None,
    ) -> tuple[ConsecutiveFlow, np.ndarray, torch.Tensor]:
        """Consume decoded chunks (``(indices, rgb uint8 [B, H, W, 3])`` on the GPU).

        Each chunk is flowed as soon as it arrives -- forward and backward for
        its pairs plus the pair bridging it to the previous chunk -- so NVDEC
        decoding (in the producer) overlaps RAFT. Returns the flows, the source
        frame index of every analysis frame, and a 1/4-size thumbnail of each
        frame (uint8, GPU) for diagnostics.
        """
        import time

        started = time.perf_counter()
        store: dict[str, torch.Tensor] = {}
        cons: list[np.ndarray] = []
        sharp: list[np.ndarray] = []
        lmean: list[np.ndarray] = []
        lstd: list[np.ndarray] = []
        thumbs: list[torch.Tensor] = []
        indices: list[np.ndarray] = []
        prev_last: torch.Tensor | None = None
        seen = 0

        def put(name: str, at: int, value: torch.Tensor) -> None:
            # Buffers grow geometrically (capacity hint first), so the final
            # result is a view, never a concatenated copy of every chunk.
            buf = store.get(name)
            need = at + len(value)
            if buf is None or len(buf) < need:
                size = max(need, capacity or 0, 2 * (len(buf) if buf is not None else 0))
                grown = torch.empty(
                    (size, *value.shape[1:]), dtype=value.dtype, device=value.device
                )
                if buf is not None:
                    grown[: len(buf)] = buf
                store[name] = buf = grown
            buf[at:need] = value

        for idx, rgb in chunks:
            indices.append(idx)
            frames = rgb if prev_last is None else torch.cat([prev_last[None], rgb])
            if len(frames) >= 2:
                a, b = frames[:-1], frames[1:]
                both = flow(torch.cat([a, b]), torch.cat([b, a]))
                f, r = both[: len(a)], both[len(a) :]
                pairs_done = seen - 1 if prev_last is not None else 0
                put("fwd", pairs_done, f.half())
                put("bwd", pairs_done, r.half())
                cons.append(consistency_mask(f, r).float().mean((1, 2)).cpu().numpy())
                del both, f, r
            sharp.append(frame_sharpness(rgb).cpu().numpy())
            luma = rgb.float().mean(-1)
            lmean.append((luma.mean((1, 2)) / 255.0).cpu().numpy())
            lstd.append((luma.std((1, 2)) / 255.0).cpu().numpy())
            thumbs.append(
                F.interpolate(rgb.permute(0, 3, 1, 2).float(), scale_factor=0.25, mode="area")
                .round()
                .to(torch.uint8)
            )
            if keep_frames:
                put("frames", seen, rgb)
            prev_last = rgb[-1]
            seen += len(rgb)
            if progress is not None:
                progress(seen)
        if seen < 2:
            raise ValueError("need at least two frames")
        result = cls(
            store["fwd"][: seen - 1],
            store["bwd"][: seen - 1],
            np.concatenate(cons),
            np.concatenate(sharp),
            np.concatenate(lmean),
            np.concatenate(lstd),
            time.perf_counter() - started,
            store["frames"][:seen] if keep_frames else None,
        )
        return result, np.concatenate(indices), torch.cat(thumbs)
