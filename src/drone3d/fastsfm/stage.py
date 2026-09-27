"""The ``sfm`` stage with ``sfm.backend: flow`` -- flow-track SfM per pass.

Each pass folder of keyframes is resized to a working resolution, tracked
with direct RAFT flow (``tracks.py``) and mapped by pycolmap. Keypoints are
written at the keyframes' *full* resolution, so the models look exactly like
spirula-studio's (``dataset/sparse/N``, cameras at image size) and every
downstream stage (georef, dense, depth, splat) reads them unchanged.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from drone3d.fastsfm.colmap import map_tracks, summarize, write_database
from drone3d.fastsfm.tracks import FlowTracks, build_tracks
from drone3d.logging_utils import get_logger

__all__ = ["load_frames", "run_flow_sfm"]

log = get_logger(__name__)


def load_frames(paths: list[Path], long_side: int, device: str = "cuda"):  # type: ignore[no-untyped-def]
    """nvJPEG-decode and area-resize keyframes -> (uint8 ``[N, h, w, 3]`` on ``device``, full (w, h))."""
    import torch
    import torch.nn.functional as F
    from torchvision.io import decode_jpeg, read_file

    out, full = [], None
    for p in paths:
        img = decode_jpeg(read_file(str(p)), device=device)
        _, h, w = img.shape
        full = full or (w, h)
        s = long_side / max(h, w)
        size = (round(h * s), round(w * s)) if s < 1 else (h, w)
        x = F.interpolate(img[None].float(), size=size, mode="area")[0] if s < 1 else img.float()
        out.append(x.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0))
    return torch.stack(out), full


def _to_full(tracks: FlowTracks, full: tuple[int, int]) -> FlowTracks:
    """Rescale observations to the full image size (corner-based pixel coordinates scale exactly)."""
    s = full[0] / tracks.size[0]
    return FlowTracks(tracks.image, tracks.track, ((tracks.xy + 0.5) * s - 0.5).astype(np.float32), full, tracks.stats)


def _components(recs: dict, min_images: int) -> list:
    """The reconstructions of one pass worth keeping, largest first."""
    return sorted((r for r in recs.values() if r.num_reg_images() >= min_images), key=lambda r: -r.num_reg_images())


def _map_pass(db: Path, images: Path, work_dir: Path, name: str, n_images: int, mapper: str, min_images: int = 3):  # type: ignore[no-untyped-def]
    """Verify and map one pass -> ``(components, mapper used, all reconstructions, timing)``.

    Every component with ``min_images`` becomes a model (Hanoi's one pass mapped
    as 43 + 25 + 6 keyframes: keeping only the largest dropped 31 registered
    views); the other mapper is tried only when all components together
    register under 80 % of the pass.
    """
    recs, t = map_tracks(db, images, work_dir / f"models_{name}", mapper=mapper)
    comps, used = _components(recs, min_images), mapper
    if sum(r.num_reg_images() for r in comps) < 0.8 * n_images:
        # the other mapper on the same verified pairs; keep whichever registers more in total
        other = "global" if mapper == "incremental" else "incremental"
        recs2, t2 = map_tracks(db, images, work_dir / f"models_{name}_{other}", mapper=other, verify=False)
        t = {**t, "mapping_s": t["mapping_s"] + t2["mapping_s"]}
        comps2 = _components(recs2, min_images)
        if sum(r.num_reg_images() for r in comps2) > sum(r.num_reg_images() for r in comps):
            comps, used, recs = comps2, other, recs2
    return comps, used, recs, t


def run_flow_sfm(
    dataset: Path,
    work_dir: Path,
    *,
    long_side: int = 960,
    span: int = 3,
    stride: int = 16,
    max_gap: int = 4,
    mapper: str = "global",
    hfov_deg: float = 72.0,
    min_images: int = 3,
    map_workers: int = 2,
) -> dict:
    """Map every pass of ``dataset/images``; models go to ``dataset/sparse/N``, largest first.

    Passes are independent: the GPU builds the tracks of one while up to
    ``map_workers`` threads verify and map the ones before it.
    """
    import shutil

    import pycolmap
    import torch

    from drone3d.engine import models

    started = time.perf_counter()
    images = dataset / "images"
    sparse = dataset / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    work_dir.mkdir(parents=True, exist_ok=True)
    passes = sorted(p for p in images.iterdir() if p.is_dir()) or [images]
    # largest first: its mapping (the longest) then overlaps the tracking of all the others
    passes.sort(key=lambda p: -sum(1 for _ in p.glob("*.jpg")))
    raft = models.raft("raft_large", batch=8, iters=12)
    per_pass, found = [], []
    timing = {"load": 0.0, "tracks": 0.0, "database": 0.0, "verification": 0.0, "mapping": 0.0}
    input_images = 0
    # GPU tracks for pass k+1 while the CPU maps pass k: pycolmap releases the GIL, so threads overlap
    pool = ThreadPoolExecutor(max_workers=map_workers)
    jobs = []
    for folder in passes:
        paths = sorted(folder.glob("*.jpg"))
        input_images += len(paths)
        if len(paths) < min_images:
            per_pass.append({"pass": folder.name, "keyframes": len(paths), "status": "too-few-keyframes"})
            continue
        t0 = time.perf_counter()
        frames, full = load_frames(paths, long_side)
        timing["load"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        tracks = _to_full(build_tracks(frames, raft, span=span, stride=stride), full)
        torch.cuda.synchronize()
        timing["tracks"] += time.perf_counter() - t0
        del frames
        names = [str(p.relative_to(images)) for p in paths]
        db = work_dir / f"{folder.name}.db"
        t0 = time.perf_counter()
        focal = 0.5 * full[0] / math.tan(math.radians(hfov_deg) / 2)
        db_info = write_database(db, images, names, tracks, focal_px=focal, max_gap=max_gap)
        timing["database"] += time.perf_counter() - t0
        row = {"pass": folder.name, "keyframes": len(paths), "tracks": tracks.stats, "database": db_info}
        jobs.append((row, pool.submit(_map_pass, db, images, work_dir, folder.name, len(paths), mapper, min_images)))
    t0 = time.perf_counter()
    for row, job in jobs:
        comps, used, recs, t = job.result()
        timing["verification"] += t["verification_s"]
        timing["mapping"] += t["mapping_s"]
        per_pass.append({**row, "mapper": used, "models": summarize(recs), "timing_s": t})
        for c, rec in enumerate(comps):  # every component of the pass with enough images is a model
            found.append((rec.num_reg_images(), row["pass"], c, rec))
    found.sort(key=lambda f: f[1])  # pass order, so equal-sized models keep a stable numbering
    pool.shutdown()
    per_pass.sort(key=lambda r: r["pass"])
    timing["mapping_wait"] = time.perf_counter() - t0  # mapping left after the last pass was tracked
    del raft
    torch.cuda.empty_cache()
    found.sort(key=lambda f: -f[0])
    models, registered, err_w = [], 0, 0.0
    for k, (n_reg, pass_name, _, rec) in enumerate(found):
        out = sparse / str(k)
        out.mkdir(parents=True, exist_ok=True)
        rec.write(str(out))
        err = float(rec.compute_mean_reprojection_error())
        models.append({"path": str(out), "images": int(n_reg), "points": int(rec.num_points3D()),
                       "mean_reprojection_px": round(err, 3), "mean_track_length": round(float(rec.compute_mean_track_length()), 2),
                       "passes": {pass_name: int(n_reg)}})  # fmt: skip
        registered += n_reg
        err_w += err * n_reg
    _ = pycolmap  # imported for its side effect of failing early when missing
    return {
        "backend": "flow",
        "workspace": str(dataset),
        "input_images": input_images,
        "registered_images": registered,
        "mean_reprojection_px": round(err_w / registered, 3) if registered else None,
        "models": models,
        "passes": per_pass,
        "settings": {"long_side": long_side, "span": span, "stride": stride, "max_gap": max_gap, "mapper": mapper},
        "stage_seconds": {k: round(v, 2) for k, v in timing.items()},
        "seconds": round(time.perf_counter() - started, 2),
    }
