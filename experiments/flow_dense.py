"""Dense mesh and point cloud from flow triangulation + TSDF, for one SfM model.

For every registered keyframe, direct RAFT flow to its ``span`` neighbours on
each side is triangulated with the model's poses (``drone3d.fastsfm.dense``),
candidate depths are fused per image, and all depth maps are integrated into
a GPU TSDF. Writes ``mesh.ply``, ``points.ply``, per-step timing and coverage.

    uv run python experiments/flow_dense.py MODEL_DIR IMAGE_DIR OUT_DIR [--long-side 480]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--long-side", type=int, default=480)
    ap.add_argument("--gaps", default="2,4,8,12", help="neighbour offsets (keyframes) to triangulate against")
    ap.add_argument("--min-angle", type=float, default=0.5, help="triangulation angle gate (deg)")
    ap.add_argument("--rel-tol", type=float, default=0.05)
    ap.add_argument("--mono", default="", help="Depth Anything V2 model id to fill untriangulated pixels")
    ap.add_argument("--voxel-px", type=float, default=2.0, help="voxel size in pixel footprints at the median depth")
    args = ap.parse_args()

    import open3d as o3d
    import pycolmap
    import torch
    import torch.nn.functional as F
    from torchvision.io import decode_jpeg, read_file

    from drone3d.fastsfm.dense import Camera, fuse_depths, pair_depth, tsdf_fuse
    from drone3d.keyframes.flow import RaftFlow, consistency_mask

    args.out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    rec = pycolmap.Reconstruction(str(args.model))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    cam0 = rec.cameras[ims[0].camera_id]
    scale = args.long_side / max(cam0.width, cam0.height)
    w, h = round(cam0.width * scale), round(cam0.height * scale)
    cams, frames = [], []
    for im in ims:
        c = Camera.from_colmap(im, rec.cameras[im.camera_id])
        cams.append(Camera(c.f * scale, c.cx * scale, c.cy * scale, c.k1, c.rotation, c.translation))
        img = decode_jpeg(read_file(str(args.images / Path(im.name).name)), device="cuda")
        x = F.interpolate(img[None].float(), size=(h, w), mode="area")[0]
        frames.append(x.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0))
    frames_t = torch.stack(frames)
    n = len(ims)
    timing["load"] = round(time.perf_counter() - t0, 2)

    # direct flow for (i, i + d), both directions
    t0 = time.perf_counter()
    raft = RaftFlow("raft_large", batch=16, iters=12)
    h8, w8 = -(-h // 8) * 8, -(-w // 8) * 8
    pad = lambda x: F.pad(x.permute(0, 3, 1, 2).float(), (0, w8 - w, 0, h8 - h), mode="replicate").round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()  # noqa: E731
    gaps = [int(g) for g in args.gaps.split(",")]
    pairs = [(i, i + d) for d in gaps for i in range(n - d)]
    cand: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(n)]
    for s in range(0, len(pairs), 32):
        chunk = pairs[s : s + 32]
        a = pad(frames_t[[p[0] for p in chunk]])
        b = pad(frames_t[[p[1] for p in chunk]])
        both = raft(torch.cat([a, b]), torch.cat([b, a]))[..., :h, :w]
        fwd, bwd = both[: len(chunk)], both[len(chunk) :]
        ok_f, ok_b = consistency_mask(fwd, bwd), consistency_mask(bwd, fwd)
        for k, (i, j) in enumerate(chunk):
            d_i, ang_i = pair_depth(cams[i], cams[j], fwd[k], ok_f[k], min_angle_deg=args.min_angle)
            d_j, ang_j = pair_depth(cams[j], cams[i], bwd[k], ok_b[k], min_angle_deg=args.min_angle)
            cand[i].append((d_i, ang_i))
            cand[j].append((d_j, ang_j))
    torch.cuda.synchronize()
    timing["flow_and_triangulation"] = round(time.perf_counter() - t0, 2)
    del raft
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    depths, coverage = [], []
    for i in range(n):
        if not cand[i]:
            depths.append(np.zeros((h, w), np.float32))
            coverage.append(0.0)
            continue
        d = torch.stack([c[0] for c in cand[i]])
        wts = torch.stack([c[1] for c in cand[i]]).clamp(max=5.0).pow(2)  # depth variance ~ 1 / angle^2
        fused = fuse_depths(d, wts, rel_tol=args.rel_tol)
        depths.append(fused.cpu().numpy())
        coverage.append(float((fused > 0).float().mean()))
    timing["fuse_depths"] = round(time.perf_counter() - t0, 2)
    mono_info = None
    if args.mono:
        from drone3d.fastsfm.mono import MonoDepth, calibrate_fill

        t0 = time.perf_counter()
        net = MonoDepth(args.mono)
        disp = net(frames_t).cpu().numpy()
        del net
        torch.cuda.empty_cache()
        timing["mono_predict"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        infos = []
        for i in range(n):
            depths[i], inf = calibrate_fill(disp[i], depths[i])
            infos.append(inf)
            coverage[i] = float((depths[i] > 0).mean())
        timing["mono_calibrate"] = round(time.perf_counter() - t0, 2)
        ok = [x for x in infos if x["status"] == "filled"]
        mono_info = {"model": args.mono, "filled_images": len(ok), "images": n,
                     "in_sample_abs_rel_median": round(float(np.median([x["in_sample_abs_rel"] for x in ok])), 4) if ok else None,
                     "statuses": sorted({x["status"] for x in infos})}
    med_depth = float(np.median(np.concatenate([d[d > 0] for d in depths])))
    voxel = args.voxel_px * med_depth / cams[0].f
    depth_max = float(np.percentile(np.concatenate([d[d > 0] for d in depths]), 99))

    np.savez_compressed(args.out / "depths.npz", depths=np.stack(depths).astype(np.float16), names=np.array([im.name for im in ims]))
    t0 = time.perf_counter()
    rgb = frames_t.cpu().numpy()
    vbg, voxel = tsdf_fuse([(depths[i], rgb[i], cams[i]) for i in range(n)], voxel=voxel, depth_max=depth_max)
    mesh = vbg.extract_triangle_mesh().to_legacy()
    pcd = vbg.extract_point_cloud().to_legacy()
    timing["tsdf_and_extract"] = round(time.perf_counter() - t0, 2)
    o3d.io.write_triangle_mesh(str(args.out / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(args.out / "points.ply"), pcd)
    report = {
        "model": str(args.model),
        "keyframes": n,
        "size": [w, h],
        "coverage_mean": round(float(np.mean(coverage)), 3),
        "coverage_min": round(float(np.min(coverage)), 3),
        "median_depth": round(med_depth, 4),
        "voxel": round(voxel, 5),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "points": len(pcd.points),
        "timing_s": timing,
        "mono": mono_info,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
