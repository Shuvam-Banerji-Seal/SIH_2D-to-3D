"""Static overlays burnt into a video: watermarks, channel logos, captions, an OSD.

They stay put while the scene moves under them. Left in, they are the one thing optical
flow tracks perfectly (zero motion on every frame), the depth prior gives them a depth,
and the texture baker stamps them onto whatever surface is behind that corner of every
keyframe. A logo is found from frames sampled across the video as sharp edges that do
not change: a clear sky or a fog bank stays put as well, but has no edges; the scene's
own edges move. Each such cluster is covered by its (padded) bounding box.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["MASK_FILE", "load_mask", "save_mask", "static_overlay_mask"]


def _pattern_agreement(stack: np.ndarray, box: tuple[slice, slice]) -> float:
    """Median over frame pairs of the normalised correlation of a box's fine detail (high-pass) -> [-1, 1].

    A burnt-in overlay repeats the same fine pattern in every frame (Mont Saint-Michel's logo: 0.83); a
    scene edge that merely stays in place -- a road the drone follows, a tower an orbit keeps centred --
    carries different detail each time (0.08-0.12).
    """
    from scipy import ndimage

    ys, xs = box
    hp = np.stack([(f - ndimage.gaussian_filter(f, 3))[ys, xs].ravel() for f in stack])
    hp -= hp.mean(1, keepdims=True)
    norm = np.linalg.norm(hp, axis=1)
    ok = norm > 1e-3
    if ok.sum() < 3:
        return 0.0
    hp = hp[ok] / norm[ok, None]
    c = hp @ hp.T
    return float(np.median(c[np.triu_indices(len(hp), 1)]))


def static_overlay_mask(luma: Sequence[np.ndarray], *, edge: float = 20.0, persist: float = 0.75, grow: float = 0.45,
                        min_moving: float = 12.0, max_share: float = 0.08, max_span: float = 0.5, min_fill: float = 0.2,
                        min_agree: float = 0.5, pad: int = 3) -> np.ndarray | None:  # fmt: skip
    """``[H, W]`` bool mask of static overlays from luma frames spread over a video, or None.

    A pixel is overlay when it lies on a sharp edge (gradient > ``edge``) in at least ``persist`` of the
    frames while the frame as a whole varies (median per-pixel range > ``min_moving``: a static shot has no
    overlay to tell apart), or in ``grow`` of them next to such a pixel (a logo's fainter lettering). Edges,
    not values: a semi-transparent watermark (Mont Saint-Michel's) changes value with the scene behind it by
    100 of 255 levels, but its outline stays. Clusters of such pixels (after closing gaps of a glyph's width)
    are replaced by their bounding boxes grown by ``pad``, unless they are not a logo's shape: a few strokes,
    thinner than two glyph widths (a letterbox edge), wider or taller than ``max_span`` of the frame (a level
    horizon), or filling under ``min_fill`` of their box, or whose fine detail does not repeat from frame to
    frame (:func:`_pattern_agreement` under ``min_agree``: a road the drone flies along, a tower an orbit keeps
    centred). None when nothing is found, the scene does not move, or the mask would cover more than
    ``max_share`` of the frame. Faint semi-transparent lettering escapes it (its detail is below the noise).
    """
    from scipy import ndimage

    if len(luma) < 6:
        return None
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in luma])
    if float(np.median(stack.max(0) - stack.min(0))) < min_moving:
        return None
    hits = np.zeros(stack.shape[1:], np.float32)
    for f in stack:
        f = ndimage.gaussian_filter(f, 0.7)
        gy, gx = np.gradient(f)
        hits += np.hypot(gx, gy) > edge
    seeds = hits >= persist * len(stack)
    h, w = seeds.shape
    glyph = max(2, round(0.004 * max(h, w)))
    weak = ndimage.binary_closing(hits >= grow * len(stack), structure=np.ones((2 * glyph + 1, 2 * glyph + 1)))
    grown, _ = ndimage.label(weak, structure=np.ones((3, 3)))
    keep = np.unique(grown[seeds & weak])
    cand = np.isin(grown, keep[keep > 0])
    labels, n = ndimage.label(cand, structure=np.ones((3, 3)))
    if not n:
        return None
    mask = np.zeros_like(cand)
    min_px = max(12, round(2e-4 * h * w))
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or int((labels[sl] == k).sum()) < min_px:
            continue
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if bw > max_span * w or bh > max_span * h or min(bh, bw) < 2 * glyph:
            continue
        if (labels[sl] == k).mean() < min_fill or _pattern_agreement(stack, sl) < min_agree:
            continue
        y0, y1 = max(0, sl[0].start - pad), min(h, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(w, sl[1].stop + pad)
        mask[y0:y1, x0:x1] = True
    if not mask.any() or mask.mean() > max_share:
        return None
    return mask


MASK_FILE = "overlay_mask.png"  # in the dataset directory, in keyframe coordinates (any size: resize to use)


def save_mask(dataset, mask: np.ndarray | None) -> None:  # type: ignore[no-untyped-def]
    """Write (or, for None, remove) the dataset's overlay mask."""
    from pathlib import Path

    from PIL import Image

    path = Path(dataset) / MASK_FILE
    if mask is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def load_mask(dataset, size: tuple[int, int] | None = None) -> np.ndarray | None:  # type: ignore[no-untyped-def]
    """The dataset's overlay mask at ``size`` = (width, height) (None: as stored), or None when the video has none."""
    from pathlib import Path

    from PIL import Image

    path = Path(dataset) / MASK_FILE
    if not path.is_file():
        return None
    with Image.open(path) as im:
        im = im.convert("L")
        return np.asarray(im if size is None else im.resize(size, Image.NEAREST)) > 127
