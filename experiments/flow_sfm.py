"""Flow-track SfM on one pass of keyframes, against spirula-studio's SIFT SfM.

Resizes the pass to a working resolution, builds tracks from direct RAFT flow
(``drone3d.fastsfm.tracks``), maps them with pycolmap (global and/or
incremental) and compares the camera poses with a reference COLMAP model of
the same images (similarity-aligned camera centres, rotation error), with the
wall time of every step.

    uv run python experiments/flow_sfm.py outputs/jal_mahal/dataset/images/pass_03 \
        outputs/jal_mahal/dataset/sparse/0 outputs/experiments/flow_sfm_pass03
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)


def load_resized(paths: list[Path], long_side: int):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F
    from torchvision.io import decode_jpeg, read_file

    frames = []
    for p in paths:
        img = decode_jpeg(read_file(str(p)), device="cuda")  # [3, H, W] uint8
        _, h, w = img.shape
        s = long_side / max(h, w)
        size = (round(h * s), round(w * s))
        x = F.interpolate(img[None].float(), size=size, mode="area")[0]
        frames.append(x.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0))
    return torch.stack(frames), (w, h)


def centres_and_rotations(rec, names: list[str], prefix: str = "") -> dict:  # type: ignore[no-untyped-def]
    out = {}
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    for n in names:
        im = by_name.get(prefix + n)
        if im is None:
            continue
        pose = im.cam_from_world()
        rot = pose.rotation.matrix()
        out[n] = (-rot.T @ pose.translation, rot)
    return out


def compare(ours: dict, ref: dict) -> dict:
    from drone3d.geo.georef import solve_similarity

    common = sorted(set(ours) & set(ref))
    if len(common) < 3:
        return {"common": len(common)}
    a = np.array([ours[n][0] for n in common])
    b = np.array([ref[n][0] for n in common])
    sim = solve_similarity(a, b)
    err = np.linalg.norm(sim.apply(a) - b, axis=1)
    extent = float(np.linalg.norm(b.max(0) - b.min(0)))
    rot_err = []
    for n in common:
        r_al = ours[n][1] @ sim.rotation.T  # world rotated into the reference frame
        d = ref[n][1] @ r_al.T
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    return {
        "common": len(common),
        "centre_rmse_rel_extent": round(float(np.sqrt((err**2).mean()) / extent), 5),
        "centre_max_rel_extent": round(float(err.max() / extent), 5),
        "rotation_err_median_deg": round(float(np.median(rot_err)), 3),
        "rotation_err_max_deg": round(float(np.max(rot_err)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("reference", type=Path, help="COLMAP model of the same images (or '-')")
    ap.add_argument("out", type=Path)
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--mappers", default="global,incremental")
    ap.add_argument("--hfov", type=float, default=72.0)
    args = ap.parse_args()

    import cv2
    import pycolmap
    import torch

    from drone3d.fastsfm.colmap import map_tracks, summarize, write_database
    from drone3d.fastsfm.tracks import FlowTracks, build_tracks
    from drone3d.keyframes.flow import RaftFlow

    args.out.mkdir(parents=True, exist_ok=True)
    img_dir = args.out / "images"
    img_dir.mkdir(exist_ok=True)
    paths = sorted(args.images.glob("*.jpg"))
    names = [p.name for p in paths]
    report: dict = {"images": str(args.images), "keyframes": len(paths), "long_side": args.long_side, "timing_s": {}}
    t0 = time.perf_counter()
    frames, src_size = load_resized(paths, args.long_side)
    torch.cuda.synchronize()
    for name, f in zip(names, frames.cpu().numpy(), strict=True):
        cv2.imwrite(str(img_dir / name), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    report["timing_s"]["load_resize_write"] = round(time.perf_counter() - t0, 2)

    cache = args.out / f"tracks_span{args.span}.npz"
    if cache.is_file():  # flow is the expensive part; mapping experiments reuse it
        z = np.load(cache, allow_pickle=True)
        tracks = FlowTracks(z["image"], z["track"], z["xy"], tuple(z["size"]), z["stats"].item())
        report["timing_s"]["tracks"] = tracks.stats.get("wall_s")
    else:
        t0 = time.perf_counter()
        raft = RaftFlow("raft_large", batch=8, iters=12)
        tracks = build_tracks(frames, raft, span=args.span)
        torch.cuda.synchronize()
        tracks.stats["wall_s"] = round(time.perf_counter() - t0, 2)
        report["timing_s"]["tracks"] = tracks.stats["wall_s"]
        np.savez(cache, image=tracks.image, track=tracks.track, xy=tracks.xy, size=np.array(tracks.size), stats=np.array(tracks.stats, dtype=object))
        del raft
        torch.cuda.empty_cache()
    report["tracks"] = tracks.stats

    w, h = tracks.size
    focal = 0.5 * w / np.tan(np.radians(args.hfov) / 2)
    ref = None
    if str(args.reference) != "-":
        ref_rec = pycolmap.Reconstruction(str(args.reference))
        prefix = args.images.name + "/"
        ref = centres_and_rotations(ref_rec, names, prefix)
        cam = next(iter(ref_rec.cameras.values()))
        report["reference"] = {"model": str(args.reference), "images": len(ref), "focal_at_working_res": round(cam.params[0] * w / src_size[0], 1)}
    report["runs"] = {}
    for mapper in args.mappers.split(","):
        db = args.out / f"{mapper}.db"
        t0 = time.perf_counter()
        db_info = write_database(db, img_dir, names, tracks, focal_px=focal)
        t_db = round(time.perf_counter() - t0, 2)
        recs, timing = map_tracks(db, img_dir, args.out / f"sparse_{mapper}", mapper=mapper)
        run = {"database": db_info, "timing_s": {"database": t_db, **timing}, "models": summarize(recs)}
        if recs and ref is not None:
            best = max(recs.values(), key=lambda r: r.num_reg_images())
            run["vs_reference"] = compare(centres_and_rotations(best, names), ref)
            run["focal_px"] = round(float(next(iter(best.cameras.values())).params[0]), 1)
        report["runs"][mapper] = run
        print(json.dumps({mapper: run}, indent=1), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("keyframes", "timing_s", "tracks")}, indent=1))


if __name__ == "__main__":
    main()
