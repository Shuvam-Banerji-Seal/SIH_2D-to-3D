"""Overlap-band keyframe selection and the per-pass "is this really 3D?" verdict.

Pipeline, all on the GPU once the analysis frames are decoded:

1. **Passes.** A cut, a fade to black or a dissolve collapses forward-backward
   flow consistency between neighbours; runs of consistent frames are passes
   (one continuous camera move each). Edited footage has several; a true single
   pass has one.
2. **Overlap band.** A grid of points on keyframe *K* is chained through the
   consecutive flows. Overlap(*K*, *t*) is the smaller of (a) the share of
   *K*'s points still tracked into *t* and (b) the share of *t* those points
   cover, so both forward and backward motion count. The next keyframe is the
   sharpest frame whose overlap lies in ``[tau, tau + delta]``: similar enough to
   match reliably, different enough to triangulate. Each surface point then
   appears in roughly ``1 / (1 - tau)`` keyframes.
3. **Verdict.** For keyframe pairs up to ``span_max`` apart, H and F are
   fitted to the chained tracks (:mod:`drone3d.keyframes.geometry`). A pass
   whose pairs prefer F under GRIC and show rotation-compensated parallax above
   ``parallax_snr`` x the track noise holds recoverable 3D structure; otherwise the camera
   only rotated, the scene is planar, or it is too far away, and geometry must
   come from a monocular prior instead.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from drone3d.keyframes.flow import ConsecutiveFlow, RaftFlow, consistency_mask
from drone3d.keyframes.geometry import two_view_analysis

__all__ = [
    "Keyframe",
    "PassReport",
    "SelectionResult",
    "SelectorConfig",
    "Tracker",
    "find_passes",
    "overlap_matrix",
    "select_keyframes",
]


@dataclass
class SelectorConfig:
    """Knobs of the selector; defaults suit 24-60 fps aerial footage."""

    overlap_target: float = 0.75  # tau: minimum co-visibility with the previous keyframe
    overlap_band: float = 0.10  # delta: candidates lie in [tau, tau + delta]
    max_gap_s: float = 2.0  # never leave more than this between keyframes
    cut_consistency: float = 0.35  # consecutive consistency below this is a cut
    dark_luma: float = 0.06  # mean luma below this is a black / fade frame
    fade_ratio: float = 0.8  # pass ends darker than this x the pass median are a fade
    flat_std: float = 0.025  # luma std below this is a blank frame
    min_pass_s: float = 2.0  # shorter runs are dropped
    grid_stride: int = 8  # tracked point spacing, analysis pixels
    cover_cell: int = 32  # coverage cell size, analysis pixels
    sharp_window_s: float = 2.0  # sharpness is judged against this rolling median
    hfov_deg: float = 72.0  # assumed horizontal FOV to express parallax as an angle
    span_max: int = 6  # keyframe pairs (i, i+k), k <= span_max, feed the verdict
    direct_min_overlap: float = 0.3  # widest partner for direct-flow geometry needs this overlap
    verdict_max_pairs: int = 24  # keyframe pairs flowed directly for the 3D verdict, per pass
    relative_overlap: bool = False  # overlap relative to the points that survive the first step
    # A pass holds recoverable 3D structure when GRIC prefers F on at least
    # this share of its widest pairs AND the rotation-compensated parallax is
    # parallax_snr times what track noise alone leaves (median over those
    # pairs; pure noise gives 1). Noise is estimated per pair from F's residuals.
    gric_3d_fraction: float = 0.6
    parallax_snr: float = 2.0
    weak_parallax_snr: float = 1.5
    sigma_floor_px: float = 0.05


@dataclass
class Keyframe:
    analysis_index: int
    frame_index: int
    timestamp_s: float
    pass_id: int
    sharpness_rel: float
    overlap_prev: float | None


@dataclass
class PassReport:
    pass_id: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    num_analysis_frames: int
    num_keyframes: int
    mean_overlap_consecutive: float | None
    views_per_point: float | None  # 1 + forward + backward mean track length (keyframes)
    pairs_analysed: int
    fraction_prefer_3d: float | None
    median_parallax_deg: float | None
    p90_parallax_deg: float | None
    parallax_snr: float | None
    verdict: str  # "3d" | "weak-3d" | "degenerate" | "too-short"
    reason: str


@dataclass
class SelectionResult:
    keyframes: list[Keyframe]
    passes: list[PassReport]
    consistency: list[float]
    sharpness: list[float]
    analysis_frame_indices: list[int]
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "passes": [asdict(p) for p in self.passes],
            "keyframes": [asdict(k) for k in self.keyframes],
            "consistency": [round(float(c), 4) for c in self.consistency],
            "sharpness": [round(float(s), 3) for s in self.sharpness],
            "analysis_frame_indices": self.analysis_frame_indices,
        }


class Tracker:
    """Chains consecutive flows to follow point sets across frames.

    Point sets are batched, ``pts [A, M, 2]`` with ``alive [A, M]``, so every
    set in flight moves with one ``grid_sample`` per frame and no host sync.
    """

    def __init__(self, flow: ConsecutiveFlow, stride: int, cell: int) -> None:
        self.flow = flow
        _, _, self.h, self.w = flow.fwd.shape
        self.device = flow.fwd.device
        ys, xs = torch.meshgrid(
            torch.arange(stride / 2, self.h, stride, device=self.device),
            torch.arange(stride / 2, self.w, stride, device=self.device),
            indexing="ij",
        )
        self.grid = torch.stack([xs.flatten(), ys.flatten()], dim=-1).float()
        self.cell = cell
        self.cells_x = math.ceil(self.w / cell)
        self.cells_y = math.ceil(self.h / cell)

    def fresh(self, count: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        pts = self.grid[None].expand(count, -1, -1).clone()
        return pts, torch.ones(pts.shape[:2], dtype=torch.bool, device=self.device)

    def _sample(self, field_: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Bilinear sample ``field_ [2, H, W]`` at ``pts [A, M, 2]`` -> ``[A, M, 2]``."""
        a, m, _ = pts.shape
        flat = pts.reshape(1, 1, a * m, 2)
        g = torch.stack(
            [2 * flat[..., 0] / (self.w - 1) - 1, 2 * flat[..., 1] / (self.h - 1) - 1], -1
        )
        out = F.grid_sample(field_[None].float(), g, align_corners=True, padding_mode="border")
        return out[0, :, 0].T.reshape(a, m, 2)

    def step(
        self, pts: torch.Tensor, alive: torch.Tensor, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move point sets from analysis frame ``t`` to ``t + 1``; drop inconsistent tracks."""
        d = self._sample(self.flow.fwd[t], pts)
        nxt = pts + d
        db = self._sample(self.flow.bwd[t], nxt)
        err = (d + db).pow(2).sum(-1)
        bound = 0.01 * (d.pow(2).sum(-1) + db.pow(2).sum(-1)) + 0.5
        inside = (
            (nxt[..., 0] >= 0)
            & (nxt[..., 0] <= self.w - 1)
            & (nxt[..., 1] >= 0)
            & (nxt[..., 1] <= self.h - 1)
        )
        return nxt, alive & (err < bound) & inside

    def overlap(self, pts: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """``[A]`` symmetric overlap: min(share of points alive, share of target cells covered).

        Scatter rather than boolean indexing, so no host sync.
        """
        a = pts.shape[0]
        cx = (pts[..., 0] / self.cell).long().clamp(0, self.cells_x - 1)
        cy = (pts[..., 1] / self.cell).long().clamp(0, self.cells_y - 1)
        cells = self.cells_x * self.cells_y
        flat = torch.arange(a, device=self.device)[:, None] * cells + cy * self.cells_x + cx
        occupied = torch.zeros(a * cells, dtype=torch.uint8, device=self.device)
        occupied.scatter_reduce_(0, flat.flatten(), alive.flatten().to(torch.uint8), reduce="amax")
        coverage = occupied.view(a, cells).float().mean(-1)
        return torch.minimum(alive.float().mean(-1), coverage)


def find_passes(flow: ConsecutiveFlow, fps: float, cfg: SelectorConfig) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` analysis-index ranges of continuous camera moves."""
    n = flow.num_frames
    bad = (flow.luma_mean < cfg.dark_luma) | (flow.luma_std < cfg.flat_std)
    cut_after = np.append(flow.consistency < cfg.cut_consistency, True)
    min_len = max(3, round(cfg.min_pass_s * fps))
    passes: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(n):
        if bad[i]:
            if start is not None and i - 1 - start + 1 >= min_len:
                passes.append((start, i - 1))
            start = None
            continue
        if start is None:
            start = i
        if cut_after[i]:
            if i - start + 1 >= min_len:
                passes.append((start, i))
            start = None
    return [
        p
        for p in (_trim_fades(flow.luma_mean, s, e, cfg.fade_ratio) for s, e in passes)
        if p[1] - p[0] + 1 >= min_len
    ]


def _trim_fades(luma: np.ndarray, s: int, e: int, ratio: float) -> tuple[int, int]:
    """Drop fade-in / fade-out frames: pass ends darker than ``ratio`` x the pass median.

    A fade is not scene content: its frames track badly (so they pile up as
    keyframes), carry no consistent exposure, and score as bad held-out views.
    """
    ref = float(np.median(luma[s : e + 1]))
    while s < e and luma[s] < ratio * ref:
        s += 1
    while e > s and luma[e] < ratio * ref:
        e -= 1
    return s, e


def _rolling_relative(values: np.ndarray, window: int) -> np.ndarray:
    half = max(1, window // 2)
    out = np.empty_like(values)
    for i in range(len(values)):
        med = np.median(values[max(0, i - half) : i + half + 1])
        out[i] = values[i] / med if med > 0 else 1.0
    return out


def overlap_matrix(tr: Tracker, s: int, e: int, horizon: int) -> np.ndarray:
    """``M[i, d - 1] = overlap(s + i, s + i + d)`` for ``d = 1..horizon`` (NaN past ``e``).

    A fresh grid starts at every frame and all live grids advance together in
    a ring of ``horizon`` slots, so the whole matrix costs one batched step per
    frame and a single read-back.
    """
    n = e - s + 1
    # Row n is a sink for slots that are empty or past the horizon.
    out = torch.full((n + 1, horizon), float("nan"), device=tr.device)
    pts, alive = tr.fresh(horizon)
    alive.zero_()
    start = torch.full((horizon,), -1, dtype=torch.long, device=tr.device)
    for t in range(s, e):
        slot = (t - s) % horizon
        pts[slot] = tr.grid
        alive[slot] = True
        start[slot] = t - s
        pts, alive = tr.step(pts, alive, t)
        ov = tr.overlap(pts, alive)
        age = (t + 1 - s) - start  # frames each slot has travelled
        ok = (start >= 0) & (age >= 1) & (age <= horizon)
        rows = torch.where(ok, start, torch.full_like(start, n))
        cols = (age - 1).clamp(0, horizon - 1)
        out[rows, cols] = ov
    return out[:n].cpu().numpy()


def _select_in_pass(
    tr: Tracker, s: int, e: int, sharp_rel: np.ndarray, fps: float, cfg: SelectorConfig
) -> tuple[list[int], list[float | None]]:
    tau, delta = cfg.overlap_target, cfg.overlap_band
    max_gap = max(2, round(cfg.max_gap_s * fps))
    matrix = overlap_matrix(tr, s, e, max_gap)
    if cfg.relative_overlap:
        # Water, reflections and moving things fail the consistency test in the first step
        # whatever the camera does; measured against what survived that step, overlap
        # counts only content that actually leaves the view (Jal Mahal pass 3 had a
        # keyframe at 2 of every 3 analysis frames because its lake died at once).
        first = np.where(matrix[:, :1] > 0.05, matrix[:, :1], np.nan)
        matrix = np.minimum(matrix / first, 1.0)
    head = min(e, s + max(1, round(0.25 * fps)))
    k = s + int(np.argmax(sharp_rel[s : head + 1]))
    chosen, overlaps = [k], [None]
    while k < e:
        stop = min(e, k + max_gap)
        ov = {u: float(matrix[k - s, u - k - 1]) for u in range(k + 1, stop + 1)}
        # Frames past the first drop below tau are not candidates, even if
        # tracking noise lifts the curve back over it later.
        first_drop = next((u for u in ov if ov[u] < tau), None)
        above = [u for u in ov if ov[u] >= tau and (first_drop is None or u < first_drop)]
        if first_drop is None and stop == e:
            # The pass ends while still overlapping K: close it with the sharpest
            # late frame unless the end view adds almost nothing new.
            if ov[e] < 0.97:
                tail = [u for u in above if u > k + (e - k) // 2] or [e]
                last = max(tail, key=lambda u: (sharp_rel[u], u))
                chosen.append(last)
                overlaps.append(ov[last])
            break
        if above:
            near = [u for u in above if ov[u] <= tau + delta] or [above[-1]]
            nxt = max(near, key=lambda u: (sharp_rel[u], u))
        else:
            nxt = k + 1  # overlap collapsed at once (fast motion): keep the chain connected
        chosen.append(nxt)
        overlaps.append(ov.get(nxt))
        k = nxt
    return chosen, overlaps


def _pair_statistics(
    tr: Tracker, kfs: list[int], cfg: SelectorConfig, flow_model: RaftFlow | None
) -> tuple[list[tuple[int, int, float]], dict, np.ndarray]:
    """Per keyframe, fit H and F against its widest well-overlapping partner.

    Chained tracks (cheap, but their error grows ~0.04 px per frame and is
    correlated) give each keyframe's overlap with the next ``span_max``
    keyframes and the multiplicity statistic. The geometry itself uses direct
    RAFT flow between the two keyframes, which stays near 0.1 px at gaps
    where chained tracks have drifted by most of a pixel -- enough drift to
    pass for parallax. Returns ``(pairs, fit, alive_frac)``.
    """
    n = len(kfs)
    span = cfg.span_max
    alive_frac = np.zeros((n, span), dtype=np.float32)
    overlap = np.zeros((n, span), dtype=np.float32)
    kf_pos = {f: i for i, f in enumerate(kfs)}
    owners: list[int] = []
    pts = torch.zeros(0, len(tr.grid), 2, device=tr.device)
    alive = torch.zeros(0, len(tr.grid), dtype=torch.bool, device=tr.device)
    pending: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for t in range(kfs[0], kfs[-1] + 1):
        j = kf_pos.get(t)
        if j is not None:
            keep = [a for a, i in enumerate(owners) if j - i <= span]
            if len(keep) != len(owners):
                pts, alive = pts[keep], alive[keep]
                owners = [owners[a] for a in keep]
            if owners:
                ov = tr.overlap(pts, alive)
                fr = alive.float().mean(-1)
                for a, i in enumerate(owners):
                    pending.append((i, j, ov[a], fr[a]))
            new_pts, new_alive = tr.fresh()
            pts, alive = torch.cat([pts, new_pts]), torch.cat([alive, new_alive])
            owners.append(j)
        if t == kfs[-1]:
            break
        pts, alive = tr.step(pts, alive, t)
    if not pending:
        return [], {}, alive_frac
    ovs = torch.stack([p_[2] for p_ in pending]).tolist()
    frs = torch.stack([p_[3] for p_ in pending]).tolist()
    for (i, j, _, _), o, f in zip(pending, ovs, frs, strict=True):
        overlap[i, j - i - 1] = o
        alive_frac[i, j - i - 1] = f
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        ok = [
            k
            for k in range(1, span + 1)
            if i + k < n and overlap[i, k - 1] >= cfg.direct_min_overlap
        ]
        if ok:
            pairs.append((i, i + ok[-1], float(overlap[i, ok[-1] - 1])))
    if len(pairs) > cfg.verdict_max_pairs:
        # The verdict is a per-pass median: evenly spaced pairs estimate it as well as all
        # of them, and each pair costs two direct RAFT flows (most of the selection time).
        pick = np.unique(np.linspace(0, len(pairs) - 1, cfg.verdict_max_pairs).round().astype(int))
        pairs = [pairs[k] for k in pick]
    if not pairs or flow_model is None or tr.flow.frames is None:
        return [], {}, alive_frac
    frames = tr.flow.frames
    a_idx = torch.tensor([kfs[i] for i, _, _ in pairs], device=frames.device)
    b_idx = torch.tensor([kfs[j] for _, j, _ in pairs], device=frames.device)
    xs, ys = tr.grid[:, 0].long(), tr.grid[:, 1].long()
    corr_x, corr_y, weights = [], [], []
    for s0 in range(0, len(pairs), 64):
        fa, fb = frames[a_idx[s0 : s0 + 64]], frames[b_idx[s0 : s0 + 64]]
        both = flow_model(torch.cat([fa, fb]), torch.cat([fb, fa]))
        fwd, bwd = both[: len(fa)], both[len(fa) :]
        ok = consistency_mask(fwd, bwd)[:, ys, xs]
        disp = fwd[:, :, ys, xs].permute(0, 2, 1)
        corr_x.append(tr.grid[None].expand(len(fa), -1, -1))
        corr_y.append(tr.grid[None] + disp)
        weights.append(ok.float())
    x, y, w = torch.cat(corr_x), torch.cat(corr_y), torch.cat(weights)
    keep = (w.sum(-1) >= 16).tolist()
    pairs = [pr for pr, k in zip(pairs, keep, strict=True) if k]
    if not pairs:
        return [], {}, alive_frac
    mask = torch.tensor(keep, device=x.device)
    focal = 0.5 * tr.w / math.tan(math.radians(cfg.hfov_deg) / 2)
    fit = two_view_analysis(
        x[mask], y[mask], w[mask], focal_px=focal, sigma=None, sigma_floor=cfg.sigma_floor_px
    )
    stats = {
        "parallax_snr": fit.parallax_snr.cpu().numpy(),
        "sigma_px": fit.sigma_px.cpu().numpy(),
        "parallax_deg": fit.parallax_deg.cpu().numpy(),
        "parallax_px": fit.parallax_px.cpu().numpy(),
        "prefers_3d": fit.prefers_3d.cpu().numpy(),
        "gric_h": fit.gric_h.cpu().numpy(),
        "gric_f": fit.gric_f.cpu().numpy(),
        "epipolar_rms_px": fit.epipolar_rms_px.cpu().numpy(),
    }
    return pairs, stats, alive_frac


def _verdict(
    pid: int, kfs: list[int], pairs: list, stats: dict, alive_frac: np.ndarray, cfg: SelectorConfig
) -> tuple[str, str, dict]:
    if len(kfs) < 3 or not pairs:
        return "too-short", f"{len(kfs)} keyframes, {len(pairs)} usable pairs", {}
    sel = np.arange(len(pairs))  # one pair per keyframe: its widest well-overlapping partner
    par = stats["parallax_deg"][sel]
    frac3d = float(np.mean(stats["prefers_3d"][sel]))
    med = float(np.nanmedian(par))
    p90 = float(np.nanpercentile(par, 90))
    fwd = alive_frac.sum(1)
    views = float(1.0 + 2.0 * np.mean(fwd[:-1])) if len(fwd) > 1 else None
    extra = {
        "fraction_prefer_3d": frac3d,
        "median_parallax_deg": med,
        "p90_parallax_deg": p90,
        "views_per_point": views,
        "pairs_analysed": len(pairs),
    }
    snr = float(np.nanmedian(stats["parallax_snr"][sel]))
    snr90 = float(np.nanpercentile(stats["parallax_snr"][sel], 90))
    extra["parallax_snr"] = snr
    extra["noise_px"] = float(np.nanmedian(stats["sigma_px"][sel]))
    ok_gric = frac3d >= cfg.gric_3d_fraction
    ok_par = snr >= cfg.parallax_snr
    detail = f"GRIC prefers F on {frac3d:.0%} of pairs, parallax {med:.2f} deg (SNR {snr:.1f})"
    # The parallax SNR decides; GRIC only confirms. On real flow
    # correspondences GRIC still prefers F on most pairs of a pure rotation
    # (F absorbs the correlated part of flow error), so it cannot veto alone.
    if ok_gric and ok_par:
        return "3d", detail, extra
    if snr >= cfg.weak_parallax_snr or snr90 >= cfg.parallax_snr:
        return "weak-3d", detail, extra
    return (
        "degenerate",
        detail + ": a homography explains the motion (rotation, planar or far-field); "
        "geometry must come from a monocular depth prior",
        extra,
    )


def select_keyframes(
    flow: ConsecutiveFlow,
    frame_indices: np.ndarray,
    fps_source: float,
    cfg: SelectorConfig | None = None,
    flow_model: RaftFlow | None = None,
) -> SelectionResult:
    """Choose keyframes over every pass of an analysed clip.

    Args:
        flow: consecutive flows of the analysis frames.
        frame_indices: source-video frame index of each analysis frame.
        fps_source: source frame rate (timestamps).
    """
    cfg = cfg or SelectorConfig()
    stride = int(frame_indices[1] - frame_indices[0]) if len(frame_indices) > 1 else 1
    fps = fps_source / stride
    tr = Tracker(flow, cfg.grid_stride, cfg.cover_cell)
    sharp_rel = _rolling_relative(flow.sharpness, max(3, round(cfg.sharp_window_s * fps)))
    keyframes: list[Keyframe] = []
    reports: list[PassReport] = []
    for pid, (s, e) in enumerate(find_passes(flow, fps, cfg)):
        chosen, overlaps = _select_in_pass(tr, s, e, sharp_rel, fps, cfg)
        for a, o in zip(chosen, overlaps, strict=True):
            keyframes.append(
                Keyframe(
                    analysis_index=int(a),
                    frame_index=int(frame_indices[a]),
                    timestamp_s=float(frame_indices[a] / fps_source),
                    pass_id=pid,
                    sharpness_rel=round(float(sharp_rel[a]), 3),
                    overlap_prev=None if o is None else round(float(o), 3),
                )
            )
        pairs, stats, alive_frac = (
            _pair_statistics(tr, chosen, cfg, flow_model)
            if len(chosen) > 1
            else ([], {}, np.zeros((0, 0)))
        )
        verdict, reason, extra = _verdict(pid, chosen, pairs, stats, alive_frac, cfg)
        consecutive = [o for o in overlaps if o is not None]
        reports.append(
            PassReport(
                pass_id=pid,
                start_frame=int(frame_indices[s]),
                end_frame=int(frame_indices[e]),
                start_s=round(float(frame_indices[s] / fps_source), 3),
                end_s=round(float(frame_indices[e] / fps_source), 3),
                num_analysis_frames=e - s + 1,
                num_keyframes=len(chosen),
                mean_overlap_consecutive=round(float(np.mean(consecutive)), 3)
                if consecutive
                else None,
                views_per_point=None
                if extra.get("views_per_point") is None
                else round(extra["views_per_point"], 2),
                pairs_analysed=int(extra.get("pairs_analysed", 0)),
                fraction_prefer_3d=None
                if "fraction_prefer_3d" not in extra
                else round(extra["fraction_prefer_3d"], 3),
                median_parallax_deg=None
                if "median_parallax_deg" not in extra
                else round(extra["median_parallax_deg"], 3),
                p90_parallax_deg=None
                if "p90_parallax_deg" not in extra
                else round(extra["p90_parallax_deg"], 3),
                parallax_snr=None
                if "parallax_snr" not in extra
                else round(extra["parallax_snr"], 2),
                verdict=verdict,
                reason=reason,
            )
        )
    return SelectionResult(
        keyframes=keyframes,
        passes=reports,
        consistency=flow.consistency.tolist(),
        sharpness=flow.sharpness.tolist(),
        analysis_frame_indices=[int(i) for i in frame_indices],
        config=asdict(cfg),
    )
