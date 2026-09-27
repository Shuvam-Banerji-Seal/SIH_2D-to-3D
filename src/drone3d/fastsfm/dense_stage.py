"""The ``dense`` stage: per-keyframe depth from flow triangulation (+ monocular fill), GPU TSDF.

For each SfM model: keyframes are resized to ``long_side``; RAFT flow to the
neighbours ``gaps`` keyframes away is triangulated with the model's poses
(wide gaps matter: consecutive keyframes of a single pass are 0.1-0.4 deg
apart at the scene, 12 apart about 1 deg); candidates are fused per image;
optionally Depth Anything V2 fills what triangulation could not reach; all
depth maps are integrated in an Open3D GPU TSDF. Writes ``mesh.ply`` and
``points.ply`` in the model frame, and per-model coverage and timing.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from drone3d.fastsfm.dense import Camera, fuse_depths, pair_depth, tsdf_fuse
from drone3d.logging_utils import get_logger

__all__ = ["run_dense"]

log = get_logger(__name__)


def _pad8(x):  # type: ignore[no-untyped-def]
    """uint8 ``[B, H, W, 3]`` -> replicate-padded to multiples of 8 (RAFT's input grid)."""
    import torch.nn.functional as F

    _, h, w, _ = x.shape
    h8, w8 = -(-h // 8) * 8, -(-w // 8) * 8
    if (h8, w8) == (h, w):
        return x
    y = F.pad(x.permute(0, 3, 1, 2).float(), (0, w8 - w, 0, h8 - h), mode="replicate")
    return y.round().to(x.dtype).permute(0, 2, 3, 1).contiguous()


def _model_frames(model_dir: Path, images: Path, long_side: int, stride: int = 1):  # type: ignore[no-untyped-def]
    import pycolmap

    from drone3d.fastsfm.stage import load_frames

    rec = pycolmap.Reconstruction(str(model_dir))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    if len(ims) >= 40:  # short passes need every view (14 keyframes at stride 2 lost 90 % of the mesh)
        ims = ims[::stride]
    else:
        stride = 1
    frames, full = load_frames([images / im.name for im in ims], long_side)
    s = frames.shape[2] / full[0]
    cams = []
    for im in ims:
        c = Camera.from_colmap(im, rec.cameras[im.camera_id])
        cams.append(Camera(c.f * s, c.cx * s, c.cy * s, c.k1, c.rotation, c.translation))
    return ims, cams, frames, stride


def run_dense(
    model_dirs: list[Path],
    images: Path,
    out_dir: Path,
    *,
    long_side: int = 480,
    gaps: tuple[int, ...] = (2, 4, 8, 12),
    keyframe_stride: int = 1,
    min_angle_deg: float = 0.5,
    rel_tol: float = 0.05,
    mono_model: str | None = "depth-anything/Depth-Anything-V2-Large-hf",
    voxel_px: float = 2.0,
) -> dict:
    import open3d as o3d
    import torch

    from drone3d.keyframes.flow import RaftFlow, consistency_mask

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    raft = RaftFlow("raft_large", batch=16, iters=12)
    mono = None
    if mono_model:
        from drone3d.fastsfm.mono import MonoDepth, calibrate_fill

        mono = MonoDepth(mono_model)
    results = []
    for model_dir in model_dirs:
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        # Depth references: every ``keyframe_stride``-th keyframe; gaps are in original keyframes.
        # Each surface is in ~4 keyframes' views, so the TSDF loses little and every step halves.
        ims, cams, frames, used_stride = _model_frames(model_dir, images, long_side, keyframe_stride)
        step_gaps = sorted({max(1, round(g / used_stride)) for g in gaps})
        n, h, w, _ = frames.shape
        timing["load"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pairs = [(i, i + d) for d in step_gaps for i in range(n - d)]
        cand: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(n)]
        for s in range(0, len(pairs), 32):
            chunk = pairs[s : s + 32]
            a, b = _pad8(frames[[p[0] for p in chunk]]), _pad8(frames[[p[1] for p in chunk]])
            both = raft(torch.cat([a, b]), torch.cat([b, a]))[..., :h, :w]
            fwd, bwd = both[: len(chunk)], both[len(chunk) :]
            ok_f, ok_b = consistency_mask(fwd, bwd), consistency_mask(bwd, fwd)
            for k, (i, j) in enumerate(chunk):
                cand[i].append(pair_depth(cams[i], cams[j], fwd[k], ok_f[k], min_angle_deg=min_angle_deg))
                cand[j].append(pair_depth(cams[j], cams[i], bwd[k], ok_b[k], min_angle_deg=min_angle_deg))
        depths = []
        for i in range(n):
            if not cand[i]:
                depths.append(np.zeros((h, w), np.float32))
                continue
            d = torch.stack([c[0] for c in cand[i]])
            wts = torch.stack([c[1] for c in cand[i]]).clamp(max=5.0).pow(2)  # depth variance ~ 1 / angle^2
            depths.append(fuse_depths(d, wts, rel_tol=rel_tol).cpu().numpy())
        torch.cuda.synchronize()
        timing["triangulate"] = time.perf_counter() - t0
        tri_cov = float(np.mean([(d > 0).mean() for d in depths]))
        fill_info = None
        if mono is not None:
            t0 = time.perf_counter()
            disp = mono(frames).cpu().numpy()
            infos = []
            for i in range(n):
                depths[i], inf = calibrate_fill(disp[i], depths[i])
                infos.append(inf)
            filled = [x for x in infos if x["status"] == "filled"]
            fill_info = {"model": mono.name, "filled_images": len(filled),
                         "in_sample_abs_rel_median": round(float(np.median([x["in_sample_abs_rel"] for x in filled])), 4) if filled else None}  # fmt: skip
            timing["mono"] = time.perf_counter() - t0
        valid = np.concatenate([d[d > 0] for d in depths]) if any((d > 0).any() for d in depths) else np.zeros(0)
        name = model_dir.name
        if not len(valid):
            results.append({"model": str(model_dir), "status": "no-depth", "keyframes": n})
            continue
        med = float(np.median(valid))
        voxel = voxel_px * med / cams[0].f
        t0 = time.perf_counter()
        rgb = frames.cpu().numpy()
        vbg, voxel = tsdf_fuse([(depths[i], rgb[i], cams[i]) for i in range(n)], voxel=voxel, depth_max=float(np.percentile(valid, 99.5)))
        mesh = vbg.extract_triangle_mesh().to_legacy()
        pcd = vbg.extract_point_cloud().to_legacy()
        timing["tsdf"] = time.perf_counter() - t0
        if not len(mesh.triangles) and not len(pcd.points):
            results.append({"model": str(model_dir), "status": "empty", "keyframes": n})
            log.info("dense %s: %d keyframes, the TSDF produced no surface", name, n)
            continue
        mdir = out_dir / f"model_{name}"
        mdir.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mdir / "mesh.ply"), mesh)
        o3d.io.write_point_cloud(str(mdir / "points.ply"), pcd)
        del frames, vbg
        torch.cuda.empty_cache()
        results.append({
            "model": str(model_dir), "status": "ok", "keyframes": n, "keyframe_stride": used_stride, "size": [w, h], "gaps": list(gaps),
            "coverage_triangulated": round(tri_cov, 4),
            "coverage": round(float(np.mean([(d > 0).mean() for d in depths])), 4),
            "voxel": round(voxel, 6), "mesh": str(mdir / "mesh.ply"), "points": str(mdir / "points.ply"),
            "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles), "num_points": len(pcd.points),
            "mono": fill_info, "timing_s": {k: round(v, 2) for k, v in timing.items()},
        })  # fmt: skip
        log.info("dense %s: %d keyframes, coverage %.2f, %d triangles", name, n, results[-1]["coverage"], len(mesh.triangles))
    del raft, mono
    torch.cuda.empty_cache()
    return {"models": results, "seconds": round(time.perf_counter() - started, 2)}
