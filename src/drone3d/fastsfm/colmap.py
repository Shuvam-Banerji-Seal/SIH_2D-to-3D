"""Feed flow tracks to COLMAP (pycolmap) and map them.

Keypoints are the tracks' positions in each image; every pair of images a
track connects within ``max_gap`` keyframes gets that correspondence as a
match. COLMAP then verifies the pairs geometrically (RANSAC) and maps them,
globally (GLOMAP) or incrementally.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from drone3d.fastsfm.tracks import FlowTracks
from drone3d.logging_utils import get_logger

__all__ = ["map_tracks", "summarize", "write_database"]

log = get_logger(__name__)


def write_database(
    db_path: Path,
    image_dir: Path,
    names: Sequence[str],
    tracks: FlowTracks,
    *,
    focal_px: float,
    camera_model: str = "SIMPLE_RADIAL",
    max_gap: int = 8,
    min_matches: int = 30,
) -> dict:
    """Create ``db_path`` with one shared camera, the track keypoints and pair matches."""
    import pycolmap

    db_path.unlink(missing_ok=True)
    pycolmap.Database.open(str(db_path)).close()  # import_images needs an existing file
    w, h = tracks.size
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = camera_model
    opts.camera_params = ",".join(f"{v:g}" for v in (focal_px, w / 2, h / 2) + ((0.0,) if "RADIAL" in camera_model else ()))
    pycolmap.import_images(str(db_path), str(image_dir), camera_mode=pycolmap.CameraMode.SINGLE,
                           image_names=list(names), options=opts)  # fmt: skip
    db = pycolmap.Database.open(str(db_path))
    try:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        n = len(names)
        kp_index = np.full(len(tracks.image), -1, dtype=np.int64)
        for i, idx in enumerate(tracks.per_image(n)):
            # COLMAP puts the centre of the top-left pixel at (0.5, 0.5).
            db.write_keypoints(ids[names[i]], (tracks.xy[idx] + 0.5).astype(np.float32))
            kp_index[idx] = np.arange(len(idx))
        order = np.lexsort((tracks.image, tracks.track))
        t, im, kp = tracks.track[order], tracks.image[order], kp_index[order]
        a_img, b_img, a_kp, b_kp = [], [], [], []
        for g in range(1, max_gap + 1):
            same = t[:-g] == t[g:]
            a_img.append(im[:-g][same])
            b_img.append(im[g:][same])
            a_kp.append(kp[:-g][same])
            b_kp.append(kp[g:][same])
        a_img, b_img = np.concatenate(a_img), np.concatenate(b_img)
        a_kp, b_kp = np.concatenate(a_kp), np.concatenate(b_kp)
        key = a_img.astype(np.int64) * n + b_img
        srt = np.argsort(key, kind="stable")
        key, a_kp, b_kp = key[srt], a_kp[srt], b_kp[srt]
        uniq, start, count = np.unique(key, return_index=True, return_counts=True)
        pairs = 0
        for k, s, c in zip(uniq, start, count, strict=True):
            if c < min_matches:
                continue
            i, j = divmod(int(k), n)
            db.write_matches(ids[names[i]], ids[names[j]], np.stack([a_kp[s : s + c], b_kp[s : s + c]], 1).astype(np.uint32))
            pairs += 1
    finally:
        db.close()
    return {"images": n, "pairs": pairs, "matches": int(count[count >= min_matches].sum())}


def map_tracks(
    db_path: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    mapper: str = "global",
    num_threads: int = 12,
    init_min_tri_angle: float = 2.0,
    min_tri_angle: float = 0.5,
    verify: bool = True,
) -> tuple[dict, dict]:
    """Verify the matched pairs and map them -> ``({model_id: Reconstruction}, timing)``."""
    import pycolmap

    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    t0 = time.perf_counter()
    if verify:
        pycolmap.geometric_verification(str(db_path))
    timing["verification_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    if mapper == "global":
        opts = pycolmap.GlobalPipelineOptions()
        opts.num_threads = num_threads
        # Dense flow tracks are far more than poses need, and global positioning + BA cost
        # grows with them: on Jal Mahal pass 3 (97 keyframes, full-resolution keypoints)
        # 2425 tracks took 26-28 s, 1200 took 10.8 s with centres 0.14 % of the extent and
        # rotations 0.28 deg (median) from SIFT SfM. No re-triangulation. ~12 tracks per image.
        db = pycolmap.Database.open(str(db_path))
        n_images = db.num_images()
        db.close()
        opts.mapper.keep_max_num_tracks = max(1000, 12 * n_images)
        opts.mapper.skip_retriangulation = True
        recs = pycolmap.global_mapping(str(db_path), str(image_dir), str(out_dir), opts)
    elif mapper == "incremental":
        opts = pycolmap.IncrementalPipelineOptions()
        opts.num_threads = num_threads
        # A single pass sees the scene through 0.1-1 deg between keyframes a few
        # apart; COLMAP's defaults (16 deg to initialise, 1.5 deg to triangulate)
        # are for photo collections and refused whole passes of our footage.
        opts.mapper.init_min_tri_angle = init_min_tri_angle
        opts.mapper.filter_min_tri_angle = min_tri_angle
        opts.triangulation.min_angle = min_tri_angle
        recs = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(out_dir), opts)
    else:
        raise ValueError(f"unknown mapper {mapper!r} (global | incremental)")
    timing["mapping_s"] = round(time.perf_counter() - t0, 2)
    return recs, timing


def summarize(recs: dict) -> list[dict]:
    """Registered images, points, reprojection error and track length per model."""
    rows = []
    for mid, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        rows.append(
            {
                "model": int(mid),
                "registered": int(rec.num_reg_images()),
                "points": int(rec.num_points3D()),
                "mean_reprojection_px": round(float(rec.compute_mean_reprojection_error()), 3),
                "mean_track_length": round(float(rec.compute_mean_track_length()), 2),
            }
        )
    return rows
