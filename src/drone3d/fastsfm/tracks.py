"""Multi-view feature tracks from dense optical flow, for structure from motion.

SIFT extraction and matching was most of the SfM time (150 of 250 s for 213
4K keyframes). The keyframe selector already runs RAFT between keyframe
pairs; this module turns that flow into tracks COLMAP can map:

* direct flow between each keyframe and its next ``span`` keyframes, both ways;
* points seeded on a grid where no track is alive, carried to the next
  keyframe by sampling the flow at their sub-pixel position (forward-backward
  consistent only);
* each step is cross-checked against the *direct* flow from one and two
  keyframes back: a track whose chained position disagrees with a direct
  prediction by more than ``max_dev_px`` ends there, and the agreeing
  predictions are averaged. Chained drift therefore cannot accumulate the way
  it does across many small steps (``experiments/flow_drift.py``).

Flow is computed in blocks of target keyframes, so memory is bounded by the
block, not by the number of keyframes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from drone3d.keyframes.flow import RaftFlow

__all__ = ["FlowTracks", "build_tracks"]


@dataclass
class FlowTracks:
    """Observations of point tracks; pixel centres at integer coordinates."""

    image: np.ndarray  # int32 [M] keyframe index of each observation
    track: np.ndarray  # int32 [M] track id
    xy: np.ndarray  # float32 [M, 2]
    size: tuple[int, int]  # (width, height) of the images the coordinates refer to
    stats: dict = field(default_factory=dict)

    def per_image(self, n: int) -> list[np.ndarray]:
        """Indices into the observation arrays, grouped by image."""
        order = np.argsort(self.image, kind="stable")
        bounds = np.searchsorted(self.image[order], np.arange(n + 1))
        return [order[bounds[i] : bounds[i + 1]] for i in range(n)]


def _sample(field_: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinear sample ``field_ [C, H, W]`` at ``pts [M, 2]`` -> ``[M, C]``."""
    _, h, w = field_.shape
    g = torch.stack([2 * pts[:, 0] / (w - 1) - 1, 2 * pts[:, 1] / (h - 1) - 1], -1)
    out = F.grid_sample(field_[None].float(), g[None, None], align_corners=True, padding_mode="border")
    return out[0, :, 0].T


def _inside(p: torch.Tensor, w: int, h: int) -> torch.Tensor:
    return (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)


def _step(fwd: torch.Tensor, bwd: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move ``pts`` along ``fwd``; forward-backward consistent and inside -> (new, ok)."""
    _, h, w = fwd.shape
    d = _sample(fwd, pts)
    nxt = pts + d
    db = _sample(bwd, nxt)
    err = (d + db).pow(2).sum(-1)
    bound = 0.01 * (d.pow(2).sum(-1) + db.pow(2).sum(-1)) + 0.5
    return nxt, (err < bound) & _inside(nxt, w, h)


class _PairFlows:
    """Direct flow for pairs ``(t - d, t)``, ``d = 1..span``, computed per block of targets."""

    def __init__(self, frames: torch.Tensor, flow: RaftFlow, span: int, block: int) -> None:
        self.frames, self.flow, self.span, self.block = frames, flow, span, block
        n, h, w, _ = frames.shape
        self.h8, self.w8 = math.ceil(h / 8) * 8, math.ceil(w / 8) * 8
        self.h, self.w = h, w
        self.cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.pairs_done = 0
        self.seconds = 0.0

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if (self.h8, self.w8) == (self.h, self.w):
            return x
        x = x.permute(0, 3, 1, 2).float()
        x = F.pad(x, (0, self.w8 - self.w, 0, self.h8 - self.h), mode="replicate")
        return x.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    def _compute(self, t0: int) -> None:
        n = len(self.frames)
        pairs = [(t - d, t) for t in range(t0, min(n, t0 + self.block)) for d in range(1, self.span + 1) if t - d >= 0]
        if not pairs:
            return
        start = time.perf_counter()
        a = torch.tensor([p[0] for p in pairs], device=self.frames.device)
        b = torch.tensor([p[1] for p in pairs], device=self.frames.device)
        fa, fb = self._pad(self.frames[a]), self._pad(self.frames[b])
        both = self.flow(torch.cat([fa, fb]), torch.cat([fb, fa]))[..., : self.h, : self.w]
        fwd, bwd = both[: len(pairs)].half(), both[len(pairs) :].half()
        for k, pair in enumerate(pairs):
            self.cache[pair] = (fwd[k], bwd[k])
        self.pairs_done += len(pairs)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - start

    def get(self, s: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (s, t) not in self.cache:
            self._compute(t)
        return self.cache[(s, t)]

    def release_before(self, t: int) -> None:
        for key in [k for k in self.cache if k[1] < t]:
            del self.cache[key]


@torch.inference_mode()
def _in_mask(mask: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Whether each ``[N, 2]`` pixel position lies in the bool ``[H, W]`` mask."""
    h, w = mask.shape
    x = xy[:, 0].round().long().clamp(0, w - 1)
    y = xy[:, 1].round().long().clamp(0, h - 1)
    return mask[y, x]


def build_tracks(
    frames: torch.Tensor,
    flow: RaftFlow,
    *,
    span: int = 3,
    stride: int = 8,
    max_dev_px: float = 1.0,
    block: int = 8,
    mask: torch.Tensor | None = None,
) -> FlowTracks:
    """Tracks through consecutive keyframes ``frames`` (uint8 ``[N, H, W, 3]`` on the GPU).

    Args:
        span: direct flow is computed from each keyframe to the next ``span``;
            ``span - 1`` of them check each chained step.
        stride: grid spacing (px) for seeding, and the occupancy cell size.
        max_dev_px: largest disagreement between the chained position and a
            direct prediction before the track ends.
        mask: bool ``[H, W]``, a burnt-in overlay (drone3d.io.overlay): it moves with the
            camera, so no track starts on it and a track that reaches it ends.
    """
    n, h, w, _ = frames.shape
    dev = frames.device
    pf = _PairFlows(frames, flow, span, block)
    ys, xs = torch.meshgrid(
        torch.arange(stride / 2, h, stride, device=dev), torch.arange(stride / 2, w, stride, device=dev), indexing="ij"
    )
    grid = torch.stack([xs.flatten(), ys.flatten()], -1).float()
    cw, ch = math.ceil(w / stride), math.ceil(h / stride)

    pos = torch.zeros(0, 2, device=dev)  # live tracks at the current keyframe
    ids = torch.zeros(0, dtype=torch.long, device=dev)
    hist = torch.full((0, span, 2), float("nan"), device=dev)  # hist[:, j]: position at keyframe k - j
    next_id = 0
    obs_img, obs_id, obs_xy = [], [], []
    ended_by_check = 0
    for k in range(n):
        # seed in empty cells, where the flow to the next keyframe is consistent
        if k < n - 1:
            occ = torch.zeros(ch * cw, dtype=torch.bool, device=dev)
            if len(pos):
                cx = (pos[:, 0] / stride).long().clamp(0, cw - 1)
                cy = (pos[:, 1] / stride).long().clamp(0, ch - 1)
                occ[cy * cw + cx] = True
            gx = (grid[:, 0] / stride).long()
            gy = (grid[:, 1] / stride).long()
            seeds = grid[~occ[gy * cw + gx]]
            if mask is not None:
                seeds = seeds[~_in_mask(mask, seeds)]
            fwd, bwd = pf.get(k, k + 1)
            _, ok = _step(fwd, bwd, seeds)
            seeds = seeds[ok]
            m = len(seeds)
            new_ids = torch.arange(next_id, next_id + m, device=dev)
            next_id += m
            pos = torch.cat([pos, seeds])
            ids = torch.cat([ids, new_ids])
            h_new = torch.full((m, span, 2), float("nan"), device=dev)
            h_new[:, 0] = seeds
            hist = torch.cat([hist, h_new])
            obs_img.append(torch.full((m,), k, dtype=torch.int32, device=dev))
            obs_id.append(new_ids.int())
            obs_xy.append(seeds)
        if k == n - 1 or not len(pos):
            continue
        # advance to k + 1: chained step, checked against direct flows from k-1, k-2, ...
        fwd, bwd = pf.get(k, k + 1)
        nxt, ok = _step(fwd, bwd, pos)
        if mask is not None:
            ok &= ~_in_mask(mask, nxt)
        acc, cnt = nxt.clone(), torch.ones(len(pos), device=dev)
        for d in range(2, span + 1):
            if k + 1 - d < 0:
                break
            prev = hist[:, d - 1]
            has = ~torch.isnan(prev[:, 0])
            if not bool(has.any()):
                continue
            f2, b2 = pf.get(k + 1 - d, k + 1)
            pred, ok2 = _step(f2, b2, torch.nan_to_num(prev))
            usable = has & ok2
            bad = usable & ((pred - nxt).norm(dim=-1) > max_dev_px)
            ended_by_check += int(bad.sum())
            ok &= ~bad
            acc += torch.where(usable[:, None], pred, 0.0)
            cnt += usable.float()
        nxt = acc / cnt[:, None]
        ok &= _inside(nxt, w, h)
        pos, ids = nxt[ok], ids[ok]
        hist = torch.cat([pos[:, None], hist[ok, : span - 1]], 1)
        obs_img.append(torch.full((len(pos),), k + 1, dtype=torch.int32, device=dev))
        obs_id.append(ids.int())
        obs_xy.append(pos)
        pf.release_before(k + 1 - span + 1)

    image = torch.cat(obs_img).cpu().numpy()
    track = torch.cat(obs_id).cpu().numpy()
    xy = torch.cat(obs_xy).cpu().numpy().astype(np.float32)
    lengths = np.bincount(track)
    keep = lengths[track] >= 2
    stats = {
        "keyframes": int(n),
        "size": [int(w), int(h)],
        "tracks": int((lengths >= 2).sum()),
        "observations": int(keep.sum()),
        "mean_track_length": round(float(lengths[lengths >= 2].mean()), 2) if (lengths >= 2).any() else 0.0,
        "ended_by_direct_check": int(ended_by_check),
        "flow_pairs": pf.pairs_done,
        "flow_seconds": round(pf.seconds, 2),
    }
    return FlowTracks(image[keep], track[keep], xy[keep], (w, h), stats)
