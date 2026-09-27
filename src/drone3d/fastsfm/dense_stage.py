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


def _release_gpu() -> None:
    """Return torch's and Open3D's cached GPU memory between models.

    Open3D keeps freed blocks of a grid in its own cache; reusing them for the
    next, differently sized grid made mesh extraction hit an illegal memory
    access (quality profile, third model of Jal Mahal) that ran fine alone.
    """
    import open3d.core as o3c
    import torch

    torch.cuda.empty_cache()
    if o3c.cuda.is_available():
        o3c.cuda.release_cache()


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


def view_coverage(vertices: np.ndarray, triangles: np.ndarray, cams: list[Camera], size: tuple[int, int],
                  sky: np.ndarray | None = None) -> float:  # fmt: skip
    """Mean share of each view's non-sky pixels whose ray hits the mesh.

    The problem statement scores completeness; this is its self-consistent
    proxy -- how much of what the camera saw was reconstructed. Exact per pixel:
    one ray per pixel is cast against the mesh (Open3D / Embree), so sparse
    vertices on near surfaces do not count as holes.
    """
    import open3d as o3d
    import open3d.core as o3c

    w, h = size
    if not len(vertices) or not len(triangles):
        return 0.0
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
                        o3c.Tensor(np.ascontiguousarray(triangles, dtype=np.uint32)))  # fmt: skip
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    shares = []
    for i, c in enumerate(cams):
        d_cam = np.stack([(xs + 0.5 - c.cx) / c.f, (ys + 0.5 - c.cy) / c.f, np.ones_like(xs)], -1).reshape(-1, 3)
        d = d_cam @ np.asarray(c.rotation, dtype=np.float32)  # R^T d as rows
        o = np.broadcast_to(np.asarray(c.centre, dtype=np.float32), d.shape)
        rays = o3c.Tensor(np.ascontiguousarray(np.concatenate([o, d], 1), dtype=np.float32))
        hit = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy()).reshape(h, w)
        valid = ~sky[i] if sky is not None else np.ones((h, w), bool)
        if valid.sum():
            shares.append(float((hit & valid).sum()) / float(valid.sum()))
    return float(np.mean(shares)) if shares else 0.0


def mesh_depth_error(vertices: np.ndarray, triangles: np.ndarray, cams: list[Camera], size: tuple[int, int],
                     ref_depths: list[np.ndarray]) -> float:  # fmt: skip
    """Median |mesh depth - reference depth| / reference depth over the reference's valid pixels.

    With the triangulated (not monocular) depth maps as reference, this is how far
    fusion moved the surface from the geometry the views actually measured.
    """
    import open3d as o3d
    import open3d.core as o3c

    w, h = size
    if not len(vertices) or not len(triangles):
        return float("nan")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
                        o3c.Tensor(np.ascontiguousarray(triangles, dtype=np.uint32)))  # fmt: skip
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    errs = []
    for c, ref in zip(cams, ref_depths, strict=True):
        m = ref > 0
        if not m.any():
            continue
        d_cam = np.stack([(xs[m] + 0.5 - c.cx) / c.f, (ys[m] + 0.5 - c.cy) / c.f, np.ones(int(m.sum()), np.float32)], -1)
        d = d_cam @ np.asarray(c.rotation, dtype=np.float32)
        o = np.broadcast_to(np.asarray(c.centre, dtype=np.float32), d.shape)
        t = scene.cast_rays(o3c.Tensor(np.ascontiguousarray(np.concatenate([o, d], 1), dtype=np.float32)))["t_hit"].numpy()
        ok = np.isfinite(t)
        if ok.any():  # d has camera-frame z = 1, so t is the depth
            errs.append(np.abs(t[ok] - ref[m][ok]) / ref[m][ok])
    return float(np.median(np.concatenate(errs))) if errs else float("nan")


def compute_depths(model_dir: Path, images: Path, raft, mono, *, long_side: int, gaps: tuple[int, ...],
                   keyframe_stride: int, min_angle_deg: float, rel_tol: float) -> dict:  # type: ignore[no-untyped-def]  # fmt: skip
    """Per-reference depth maps of one model: flow triangulation, then the monocular fill.

    Returns a dict with ``ims, cams, frames`` (uint8 on the GPU), ``depths``, ``sky``,
    ``used_stride``, ``tri_cov``, ``fill_info`` and ``timing``.
    """
    import torch

    from drone3d.fastsfm.mono import calibrate_fill
    from drone3d.keyframes.flow import consistency_mask

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
    tri_depths = [d.copy() for d in depths]  # triangulated only, before the monocular fill
    fill_info, sky = None, None
    if mono is not None:
        t0 = time.perf_counter()
        disp = mono(frames).cpu().numpy()
        sky = disp <= 0.005 * np.maximum(disp.reshape(len(disp), -1).max(1), 1e-6)[:, None, None]
        infos = []
        for i in range(n):
            depths[i], inf = calibrate_fill(disp[i], depths[i])
            infos.append(inf)
        filled = [x for x in infos if x["status"] == "filled"]
        fill_info = {"model": mono.name, "filled_images": len(filled),
                     "in_sample_abs_rel_median": round(float(np.median([x["in_sample_abs_rel"] for x in filled])), 4) if filled else None}  # fmt: skip
        timing["mono"] = time.perf_counter() - t0
    return {"ims": ims, "cams": cams, "frames": frames, "depths": depths, "tri_depths": tri_depths, "sky": sky,
            "used_stride": used_stride,
            "tri_cov": tri_cov, "fill_info": fill_info, "timing": timing}  # fmt: skip


def _dense_model(model_dir: Path, images: Path, out_dir: Path, raft, mono, *, long_side: int,
                 gaps: tuple[int, ...], keyframe_stride: int, min_angle_deg: float, rel_tol: float,
                 voxel_px: float, trunc_voxels: float = 12.0, tsdf_memory_gb: float = 8.0) -> dict:  # type: ignore[no-untyped-def]  # fmt: skip
    """Depth, fusion and mesh for one SfM model -> its result record."""
    import open3d as o3d

    r = compute_depths(model_dir, images, raft, mono, long_side=long_side, gaps=gaps, keyframe_stride=keyframe_stride,
                       min_angle_deg=min_angle_deg, rel_tol=rel_tol)  # fmt: skip
    cams, frames, depths, sky, timing = r["cams"], r["frames"], r["depths"], r["sky"], r["timing"]
    used_stride, tri_cov, fill_info = r["used_stride"], r["tri_cov"], r["fill_info"]
    n, h, w, _ = frames.shape
    valid = np.concatenate([d[d > 0] for d in depths]) if any((d > 0).any() for d in depths) else np.zeros(0)
    name = model_dir.name
    if not len(valid):
        return {"model": str(model_dir), "status": "no-depth", "keyframes": n}
    med = float(np.median(valid))
    voxel = voxel_px * med / cams[0].f
    t0 = time.perf_counter()
    rgb = frames.cpu().numpy()
    vbg, voxel = tsdf_fuse([(depths[i], rgb[i], cams[i]) for i in range(n)], voxel=voxel,
                           depth_max=float(np.percentile(valid, 99.5)), trunc_voxels=trunc_voxels,
                           memory_gb=tsdf_memory_gb)  # fmt: skip
    log.info("dense %s: TSDF voxel %.4g, %d active blocks of %d", name, voxel, int(vbg.hashmap().size()),
             int(vbg.hashmap().capacity()))  # fmt: skip
    mesh = vbg.extract_triangle_mesh().to_legacy()
    pcd = vbg.extract_point_cloud().to_legacy()
    timing["tsdf"] = time.perf_counter() - t0
    del frames, vbg
    _release_gpu()
    if not len(mesh.triangles) and not len(pcd.points):
        log.info("dense %s: %d keyframes, the TSDF produced no surface", name, n)
        return {"model": str(model_dir), "status": "empty", "keyframes": n}
    completeness = view_coverage(np.asarray(mesh.vertices), np.asarray(mesh.triangles), cams, (w, h), sky)
    mdir = out_dir / f"model_{name}"
    mdir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mdir / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(mdir / "points.ply"), pcd)
    rec = {
        "model": str(model_dir), "status": "ok", "keyframes": n, "keyframe_stride": used_stride, "size": [w, h], "gaps": list(gaps),
        "coverage_triangulated": round(tri_cov, 4),
        "coverage": round(float(np.mean([(d > 0).mean() for d in depths])), 4),
        "view_completeness": round(completeness, 4),
        "voxel": round(voxel, 6), "mesh": str(mdir / "mesh.ply"), "points": str(mdir / "points.ply"),
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles), "num_points": len(pcd.points),
        "mono": fill_info, "timing_s": {k: round(v, 2) for k, v in timing.items()},
    }  # fmt: skip
    log.info("dense %s: %d keyframes, coverage %.2f, completeness %.2f, %d triangles",
             name, n, rec["coverage"], completeness, len(mesh.triangles))  # fmt: skip
    return rec


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
    voxel_px: float = 3.0,
    trunc_voxels: float = 12.0,
    tsdf_memory_gb: float = 8.0,
) -> dict:
    import torch

    from drone3d.engine import models

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    raft = models.raft("raft_large", batch=16, iters=12)
    mono = None
    if mono_model:
        mono = models.mono(mono_model)
    results = []
    for model_dir in model_dirs:
        try:
            results.append(_dense_model(model_dir, images, out_dir, raft, mono, long_side=long_side, gaps=gaps,
                                        keyframe_stride=keyframe_stride, min_angle_deg=min_angle_deg, rel_tol=rel_tol,
                                        voxel_px=voxel_px, trunc_voxels=trunc_voxels,
                                        tsdf_memory_gb=tsdf_memory_gb))  # fmt: skip
        except RuntimeError as exc:  # a CUDA / Open3D failure on one model must not lose the others
            log.warning("dense %s failed: %s", model_dir.name, str(exc)[:300])
            results.append({"model": str(model_dir), "status": "failed", "error": str(exc)[:300]})
            torch.cuda.empty_cache()
    del raft, mono
    torch.cuda.empty_cache()
    return {"models": results, "seconds": round(time.perf_counter() - started, 2)}
