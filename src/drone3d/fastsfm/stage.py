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
from drone3d.io.overlay import load_mask
from drone3d.logging_utils import get_logger

__all__ = ["load_frames", "run_flow_sfm"]

log = get_logger(__name__)


def load_frames(paths: list[Path], long_side: int, device: str = "cuda"):  # type: ignore[no-untyped-def]
    """nvJPEG-decode and area-resize keyframes -> (uint8 ``[N, h, w, 3]`` on ``device``, full (w, h))."""
    import torch
    import torch.nn.functional as F
    from torchvision.io import read_file

    from drone3d.gpu.nvjpeg import decode_jpeg

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


PLAUSIBLE_HFOV = (25.0, 130.0)  # degrees: telephoto-ish cinematic to FPV action cameras


def _hfov(cam) -> float:  # type: ignore[no-untyped-def]
    return math.degrees(2 * math.atan(cam.width / 2 / cam.params[0]))


def _consensus_intrinsics(found: list, images: Path, work_dir: Path, mapper: str, min_images: int,
                          timing: dict) -> tuple[list, dict]:  # fmt: skip
    """Re-map the passes whose self-calibration ran away, with the video's consensus focal length and k1.

    A single pass constrains focal length and radial distortion weakly, and bundle adjustment wandered: of 114
    cameras on the sample videos, Eiffel's main model came out at a 110 deg field of view, others at 8 deg
    with k1 = 27, or 178 deg -- geometry sheared and domed accordingly. One video is usually one camera, so
    the keyframe-weighted median over the plausible passes is its focal length. A pass is mapped again with
    the consensus held fixed when its own solution is implausible (field of view outside PLAUSIBLE_HFOV,
    |k1| > 0.3), or when the consensus holds >= 60 % of the keyframes and the pass is > 1.5x off it --
    cinematic drones switch lenses, so a moderate disagreement can be real. The re-map is kept if it registers
    >= 80 % as many keyframes at under 1.5 px of reprojection error.
    """
    cams = [(n, p, next(iter(rec.cameras.values()))) for n, p, _, rec in found]
    good = [(n, c) for n, _, c in cams if PLAUSIBLE_HFOV[0] <= _hfov(c) <= PLAUSIBLE_HFOV[1] and abs(c.params[3] if len(c.params) > 3 else 0) <= 0.3]
    if not good or sum(n for n, _ in good) < 0.3 * sum(n for n, _, _ in cams):
        return found, {"status": "no-consensus"}
    order = sorted(good, key=lambda g: g[1].params[0])
    w = np.cumsum([n for n, _ in order])
    focal = float(order[int(np.searchsorted(w, w[-1] / 2))][1].params[0])  # weighted median
    ks = sorted((c.params[3] if len(c.params) > 3 else 0.0, n) for n, c in good)
    wk = np.cumsum([n for _, n in ks])
    k1 = float(ks[int(np.searchsorted(wk, wk[-1] / 2))][0])
    strong = sum(n for n, c in good if abs(c.params[0] / focal - 1) < 0.25) >= 0.6 * sum(n for n, _, _ in cams)
    t0 = time.perf_counter()
    redone, out = [], []
    for entry in found:
        n_reg, pass_name, comp, rec = entry
        cam = next(iter(rec.cameras.values()))
        implausible = not PLAUSIBLE_HFOV[0] <= _hfov(cam) <= PLAUSIBLE_HFOV[1] or abs(cam.params[3] if len(cam.params) > 3 else 0) > 0.3
        off = implausible or (strong and not 1 / 1.5 <= cam.params[0] / focal <= 1.5)
        if not off or comp != 0:  # one re-map per pass: its first component carries the decision
            out.append(entry)
            continue
        db = work_dir / f"{pass_name}.db"
        try:
            recs, _ = map_tracks(db, images, work_dir / f"models_{pass_name}_consensus", mapper=mapper, verify=False,
                                 fixed_intrinsics=(focal, k1))  # fmt: skip
        except Exception as exc:  # the original mapping stands
            log.warning("sfm: re-mapping %s with the consensus intrinsics failed: %s", pass_name, str(exc)[:160])
            out.append(entry)
            continue
        comps = _components(recs, min_images)
        if comps and comps[0].num_reg_images() >= 0.8 * n_reg and comps[0].compute_mean_reprojection_error() < 1.5:
            out.append((comps[0].num_reg_images(), pass_name, 0, comps[0]))
            redone.append({"pass": pass_name, "hfov_before": round(_hfov(cam), 1), "k1_before": round(float(cam.params[3]) if len(cam.params) > 3 else 0.0, 3),
                           "registered_before": n_reg, "registered_after": comps[0].num_reg_images()})  # fmt: skip
        else:
            out.append(entry)
    timing["intrinsics"] = time.perf_counter() - t0
    info = {"status": "ok", "strong": strong, "focal_px": round(focal, 1), "k1": round(k1, 4), "hfov_deg": round(math.degrees(2 * math.atan(cams[0][2].width / 2 / focal)), 1),
            "remapped": redone}  # fmt: skip
    if redone:
        log.info("sfm: consensus intrinsics (hfov %.0f deg, k1 %.3f) re-mapped %d pass(es)", info["hfov_deg"], k1, len(redone))
    return out, info


def _merge(dataset: Path, models: list[dict], pairs: int, timing: dict) -> tuple[list[dict], dict]:
    """Merge the passes' models that see the same scene (see :mod:`drone3d.fastsfm.merge`) -> (models, info).

    The per-pass models move to ``dataset/sparse_passes``; ``dataset/sparse/N`` become the merged ones.
    A failure keeps the per-pass models: merging only ever adds to a run.
    """
    import shutil

    import pycolmap

    from drone3d.engine import models as engine_models
    from drone3d.fastsfm.merge import merge_models

    sparse, passes_dir = dataset / "sparse", dataset / "sparse_passes"
    t0 = time.perf_counter()
    try:
        if passes_dir.exists():
            shutil.rmtree(passes_dir)
        shutil.copytree(sparse, passes_dir)
        by_name = {Path(m["path"]).name: m for m in models}
        with engine_models.gpu_exclusive():
            out = merge_models([passes_dir / n for n in by_name], dataset / "images", dataset / "sparse_merged", pairs_per_model_pair=pairs)
    except Exception as exc:  # RoMa missing, out of memory, ...: the per-pass models stand
        log.warning("merging passes failed (%s); keeping the per-pass models", str(exc)[:200])
        shutil.rmtree(passes_dir, ignore_errors=True)
        return models, {"status": "failed", "error": str(exc)[:300]}
    merged = []
    shutil.rmtree(sparse)
    for k, g in enumerate(out["groups"]):
        dst = sparse / str(k)
        shutil.move(g["path"], dst)
        rec = pycolmap.Reconstruction(str(dst))
        passes: dict[str, int] = {}
        for name in g["members"]:
            for p_, n in by_name[name]["passes"].items():
                passes[p_] = passes.get(p_, 0) + n
        merged.append({"path": str(dst), "images": int(rec.num_reg_images()), "points": int(rec.num_points3D()),
                       "mean_reprojection_px": round(float(rec.compute_mean_reprojection_error()), 3),
                       "mean_track_length": round(float(rec.compute_mean_track_length()), 2),
                       "passes": passes, "merged_from": g["members"]})  # fmt: skip
    shutil.rmtree(dataset / "sparse_merged", ignore_errors=True)
    timing["merge"] = time.perf_counter() - t0
    info = {"status": "ok", "models_before": len(models), "models_after": len(merged), "edges": out["edges"], "timing": out["timing"]}
    log.info("sfm: %d pass models merged into %d", len(models), len(merged))
    return merged, info


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
    merge: bool = True,
    merge_pairs: int = 4,
    consensus: bool = True,
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
        with models.gpu_exclusive():  # a multi-slot engine: the tracking holds the GPU, the mapping below does not
            t0 = time.perf_counter()
            frames, full = load_frames(paths, long_side)
            timing["load"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            overlay = load_mask(images.parent, (frames.shape[2], frames.shape[1]))
            overlay = torch.from_numpy(overlay).to(frames.device) if overlay is not None else None
            tracks = _to_full(build_tracks(frames, raft, span=span, stride=stride, mask=overlay), full)
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
    found, intrinsics = _consensus_intrinsics(found, images, work_dir, mapper, min_images, timing) if consensus else (found, None)
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
    merge_info = None
    if merge and len(models) > 1:
        models, merge_info = _merge(dataset, models, merge_pairs, timing)
    return {
        "backend": "flow",
        "workspace": str(dataset),
        "input_images": input_images,
        "registered_images": registered,
        "mean_reprojection_px": round(err_w / registered, 3) if registered else None,
        "models": models,
        "passes": per_pass,
        "merge": merge_info,
        "intrinsics": intrinsics,
        "settings": {"long_side": long_side, "span": span, "stride": stride, "max_gap": max_gap, "mapper": mapper},
        "stage_seconds": {k: round(v, 2) for k, v in timing.items()},
        "seconds": round(time.perf_counter() - started, 2),
    }
