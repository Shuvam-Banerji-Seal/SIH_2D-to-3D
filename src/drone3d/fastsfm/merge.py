"""Put the per-pass SfM models of one video into one frame: RoMa v2 across passes, Sim(3) between models.

The fast profile maps each pass (shot) on its own, so an edit that circles a building in five shots gives five
models -- and the back of the building, seen only in another shot, is missing from each. For every pair of
models this retrieves the keyframe pairs most likely to overlap (cosine similarity of mean-pooled DINOv3
features, RoMa v2's own backbone), matches them densely with RoMa v2, lifts a match to 3D where it lands next
to an SfM observation in both images, and fits a similarity in RANSAC. Models joined by confident
similarities are written as one reconstruction in the frame of the largest: the dense stage then fuses every
pass's depth into one TSDF, and the texture and the splats see every camera.

Colosseum (11 shots, 75 keyframes): 7 models (52 keyframes) became one, 2 more another; residuals 1-2 % of
the scene (``experiments/merge_passes.py``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["compose", "invert", "merge_models", "ransac_sim3", "umeyama"]

log = get_logger(__name__)

Sim3 = tuple[float, np.ndarray, np.ndarray]  # s, R, t: x -> s R x + t


def umeyama(a: np.ndarray, b: np.ndarray) -> Sim3:
    """The similarity minimising ``|s R a + t - b|`` (Umeyama 1991)."""
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = a - ma, b - mb
    u, d, vt = np.linalg.svd(cb.T @ ca / len(a))
    fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        fix[2, 2] = -1
    r = u @ fix @ vt
    s = float(np.trace(np.diag(d) @ fix) / max(float(ca.var(0).sum()), 1e-12))
    return s, r, mb - s * r @ ma


def compose(p: Sim3, q: Sim3) -> Sim3:
    """``p o q``: first q, then p."""
    return p[0] * q[0], p[1] @ q[1], p[0] * p[1] @ q[2] + p[2]


def invert(p: Sim3) -> Sim3:
    s, r, t = p
    return 1.0 / s, r.T, -(r.T @ t) / s


def _umeyama_batch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`umeyama` for ``[K, m, 3]`` point sets at once -> ``s [K], R [K, 3, 3], t [K, 3]``."""
    ma, mb = a.mean(1), b.mean(1)
    ca, cb = a - ma[:, None], b - mb[:, None]
    u, d, vt = np.linalg.svd(np.einsum("kni,knj->kij", cb, ca) / a.shape[1])
    fix = np.ones((len(a), 3))
    fix[:, 2] = np.where(np.linalg.det(u) * np.linalg.det(vt) < 0, -1.0, 1.0)
    r = u @ (fix[:, :, None] * vt)
    s = (d * fix).sum(1) / np.maximum((ca**2).sum((1, 2)) / a.shape[1], 1e-12)
    return s, r, mb - s[:, None] * np.einsum("kij,kj->ki", r, ma)


def ransac_sim3(a: np.ndarray, b: np.ndarray, thresh: float, *, iters: int = 2000, seed: int = 0,
                chunk: int = 512) -> dict | None:  # fmt: skip
    """Robust similarity taking ``a`` onto ``b`` (``[N, 3]`` each) -> fit with inlier count and residual, or None.

    Every hypothesis is fitted in one batched SVD and scored ``chunk`` at a time, on the GPU when there is
    one: one at a time in numpy, 2000 per model pair were 44 of the 83 s Colosseum's merge took.
    """
    if len(a) < 8:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (iters, 4))
    idx = idx[(np.diff(np.sort(idx, axis=1), axis=1) > 0).all(1)]  # four distinct points
    s, r, t = _umeyama_batch(a[idx], b[idx])
    ok = np.isfinite(s) & (s > 0) & np.isfinite(r).all((1, 2)) & np.isfinite(t).all(1)
    m = s[ok, None, None] * r[ok]  # s R
    t = t[ok]
    if not len(m):
        return None
    counts = np.zeros(len(m), dtype=np.int64)
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else None
    except ImportError:
        dev = None
    if dev:
        ta, tb = torch.as_tensor(a, device=dev), torch.as_tensor(b, device=dev)
        for c in range(0, len(m), chunk):
            d = ta[None] @ torch.as_tensor(m[c : c + chunk], device=dev).transpose(1, 2)
            d += torch.as_tensor(t[c : c + chunk], device=dev)[:, None] - tb[None]
            counts[c : c + chunk] = ((d * d).sum(2) < thresh**2).sum(1).cpu().numpy()
    else:
        for c in range(0, len(m), chunk):
            d = a[None] @ m[c : c + chunk].transpose(0, 2, 1) + t[c : c + chunk, None] - b[None]
            counts[c : c + chunk] = (np.einsum("kni,kni->kn", d, d) < thresh**2).sum(1)
    k = int(np.argmax(counts))
    if counts[k] < 8:
        return None
    best = np.linalg.norm(a @ m[k].T + t[k] - b, axis=1) < thresh
    s, r, t = umeyama(a[best], b[best])
    res = np.linalg.norm(s * a @ r.T + t - b, axis=1)
    inl = res < thresh
    return {"sim3": (s, r, t), "inliers": int(inl.sum()), "candidates": len(a), "median_residual": float(np.median(res[inl]))}


_ROMA: dict[str, Any] = {}


def _roma(setting: str):  # type: ignore[no-untyped-def]
    """RoMa v2 (MIT; its checkpoint carries its DINOv3 backbone), loaded once per process."""
    if setting not in _ROMA:
        import os

        os.environ.setdefault("TORCH_HOME", "/store/huggingface/torch")
        from romav2 import RoMaV2

        _ROMA.clear()
        _ROMA[setting] = RoMaV2(RoMaV2.Cfg(setting=setting)).cuda().eval()
    return _ROMA[setting]


LIFT_SIZE = (640, 360)  # the calibrated depth maps that lift matches to 3D


def _calibrated_depth(rec, im, images: Path, mono) -> tuple[np.ndarray, float] | None:  # type: ignore[no-untyped-def]
    """The prior's depth for one keyframe, calibrated on its SfM tie points (the dense stage's fill) -> (depth, scale)."""
    import torch
    from PIL import Image

    from drone3d.fastsfm.dense_stage import tie_depths
    from drone3d.fastsfm.mono import calibrate_fill

    cam = rec.cameras[im.camera_id]
    w, h = LIFT_SIZE
    scale = w / cam.width
    rgb = np.asarray(Image.open(images / im.name).convert("RGB").resize((w, round(cam.height * scale))))
    disp = mono(torch.from_numpy(rgb)[None].cuda())[0].cpu().numpy()
    (tie,) = tie_depths(rec, [im], scale, (rgb.shape[1], rgb.shape[0]))
    depth, info = calibrate_fill(disp, tie, min_samples=60, sky_rel=getattr(mono, "sky_rel", 0.005))
    return (depth, scale) if info.get("status") == "filled" else None


def _lift(rec, im, xy: np.ndarray, dm: tuple[np.ndarray, float] | None) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """Full-resolution pixels -> world points where the calibrated depth has a value -> (points, mask)."""
    if dm is None:
        return np.zeros((0, 3)), np.zeros(len(xy), bool)
    depth, scale = dm
    cam = rec.cameras[im.camera_id]
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    u = np.clip((xy[:, 0] * scale).astype(int), 0, depth.shape[1] - 1)
    v = np.clip((xy[:, 1] * scale).astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]
    ok = z > 0
    xc = np.stack([(xy[ok, 0] - cx) / f * z[ok], (xy[ok, 1] - cy) / f * z[ok], z[ok]], 1)
    pose = im.cam_from_world()
    r, t = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
    return (xc - t) @ r, ok  # R^T (x_cam - t)


MATCH_BATCH = 16  # keyframe pairs per RoMa forward: 283 -> 97 ms per pair on the A100, same warps (0.01 px median)


def _match_batched(roma, pairs: list[tuple[Any, Any]], matches_per_pair: int) -> list[Any]:  # type: ignore[no-untyped-def]
    """RoMa v2 ("fast": 512^2, one direction) on ``(A, B)`` pairs of uint8 [1, 3, 512, 512] tensors -> sampled matches."""
    import torch
    from romav2.romav2 import _map_confidence

    out = []
    with torch.inference_mode():
        for s in range(0, len(pairs), MATCH_BATCH):
            chunk = pairs[s : s + MATCH_BATCH]
            pred = roma(torch.cat([a for a, _ in chunk]).float() / 255, torch.cat([b for _, b in chunk]).float() / 255)
            for i in range(len(chunk)):
                conf = pred["confidence_AB"][i : i + 1]
                overlap, precision = _map_confidence(confidence=conf, threshold=roma.threshold)
                one = {"warp_AB": pred["warp_AB"][i : i + 1], "overlap_AB": overlap, "precision_AB": precision}
                out.append(roma.sample(one, matches_per_pair)[0])
    return out


def find_links(recs: dict[str, Any], images: Path, *, pairs_per_model_pair: int = 4, matches_per_pair: int = 3000,
               setting: str = "fast", mono=None, max_partners: int = 6) -> tuple[list[dict], dict]:  # type: ignore[no-untyped-def]  # fmt: skip
    """Similarities between pairs of models -> (edges, timing). An edge maps ``b``'s frame onto ``a``'s.

    Every keyframe is decoded once (nvJPEG) and kept at RoMa's 512^2 on the GPU as uint8 (0.75 MB a keyframe);
    the per-pair path decoded both JPEGs with PIL for every pair, so a keyframe in eight pairs was decoded
    eight times. The pairs are matched in batches (``MATCH_BATCH``).

    Every pair of models is matched up to ``max_partners + 1`` models; beyond, a pair is matched when either
    model is among the other's ``max_partners`` most similar (best keyframe-pair cosine), so the work grows
    linearly, not quadratically, with the passes of a long edit (60 passes: 1770 pairs of models against at
    most 360). The global descriptor ranks true links poorly -- as low as the 6th most similar partner on
    Colosseum -- hence a wide bound; each edge records its ``rank``.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.io import read_file

    from drone3d.engine import models
    from drone3d.gpu.nvjpeg import decode_jpeg

    timing = {"load": 0.0, "descriptors": 0.0, "matching": 0.0, "lifting": 0.0}
    mono = mono or models.mono()
    depth_cache: dict[tuple[str, int], Any] = {}
    t0 = time.perf_counter()
    roma = _roma(setting)
    timing["load"] = time.perf_counter() - t0
    keys, small, lr = [], [], {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        for name, rec in recs.items():
            for im in rec.images.values():
                if not im.has_pose:
                    continue
                x = decode_jpeg(read_file(str(images / im.name)), device="cuda").float()[None] / 255
                lr[im.name] = F.interpolate(x, size=(roma.H_lr, roma.W_lr), mode="bicubic", align_corners=False,
                                            antialias=True).clamp(0, 1).mul(255).round().to(torch.uint8)  # fmt: skip
                small.append(F.interpolate(x, size=(256, 448), mode="bicubic", align_corners=False, antialias=True))
                keys.append((name, im.image_id, im.name))
        # a global descriptor per keyframe: RoMa's DINOv3 features of the deeper layer, mean-pooled
        desc = [F.normalize(roma.f(torch.cat(small[s : s + 32]))[-1].float().mean(dim=(1, 2)), dim=1).cpu()
                for s in range(0, len(small), 32)]  # fmt: skip
    del small
    timing["descriptors"] = time.perf_counter() - t0
    if len(keys) < 2:
        return [], timing
    d = torch.cat(desc).numpy()
    sim = d @ d.T
    model_of = np.array([k[0] for k in keys])
    names = sorted(recs, key=lambda k: -recs[k].num_reg_images())
    idx = {n: np.where(model_of == n)[0] for n in names}
    best = {(a, b): float(sim[np.ix_(idx[a], idx[b])].max()) for a in names for b in names if a != b}
    rank = {}  # (a, b) -> b's place among a's partners by similarity (0: the most similar)
    for a in names:
        for r_, b in enumerate(sorted((b for b in names if b != a), key=lambda b: -best[(a, b)])):
            rank[(a, b)] = r_
    todo = []  # (a, b, [(key_a, key_b), ...]) for every pair of models matched: its most similar keyframe pairs
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            place = min(rank[(a, b)], rank[(b, a)])
            if len(names) > max_partners + 1 and place >= max_partners:
                continue
            block = sim[np.ix_(idx[a], idx[b])]
            order = np.dstack(np.unravel_index(np.argsort(-block, axis=None), block.shape))[0][:pairs_per_model_pair]
            todo.append((a, b, [(keys[idx[a][r_]], keys[idx[b][c_]]) for r_, c_ in order]))
    timing["model_pairs"] = float(len(todo))
    t0 = time.perf_counter()
    flat = [(lr[ka[2]], lr[kb[2]]) for _, _, kp in todo for ka, kb in kp]
    sampled = iter(_match_batched(roma, flat, matches_per_pair))
    torch.cuda.synchronize()
    timing["matching"] = time.perf_counter() - t0
    del lr, flat
    edges = []
    for a, b, kp in todo:
        pa, pb = [], []
        for ka, kb in kp:
            matches = next(sampled)
            ima, imb = recs[a].images[ka[1]], recs[b].images[kb[1]]
            ca, cb = recs[a].cameras[ima.camera_id], recs[b].cameras[imb.camera_id]
            m = matches.float().cpu().numpy()  # [-1, 1] normalised coordinates in each image
            xa = np.stack([(m[:, 0] + 1) / 2 * ca.width, (m[:, 1] + 1) / 2 * ca.height], 1)
            xb = np.stack([(m[:, 2] + 1) / 2 * cb.width, (m[:, 3] + 1) / 2 * cb.height], 1)
            t1 = time.perf_counter()
            for key, rec, im in (((a, ima.image_id), recs[a], ima), ((b, imb.image_id), recs[b], imb)):
                if key not in depth_cache:
                    depth_cache[key] = _calibrated_depth(rec, im, images, mono)
            Xa, oka = _lift(recs[a], ima, xa, depth_cache[(a, ima.image_id)])
            Xb, okb = _lift(recs[b], imb, xb, depth_cache[(b, imb.image_id)])
            timing["lifting"] += time.perf_counter() - t1
            both = oka & okb  # a match lifts only where both keyframes have depth
            if both.any():
                pa.append(Xa[both[oka]])
                pb.append(Xb[both[okb]])
        if not pa or sum(len(x) for x in pa) < 12:
            continue
        A, B = np.concatenate(pa), np.concatenate(pb)
        if len(A) > 4000:  # plenty: a subset keeps RANSAC fast
            pick = np.random.default_rng(0).choice(len(A), 4000, replace=False)
            A, B = A[pick], B[pick]
        scale = float(np.median(np.linalg.norm(A - A.mean(0), axis=1))) or 1.0
        fit = ransac_sim3(B, A, thresh=0.03 * scale)  # b's points onto a's
        if fit is None:
            continue
        edges.append({"a": a, "b": b, "sim3": fit["sim3"], "inliers": fit["inliers"], "candidates": fit["candidates"],
                      "residual_rel": fit["median_residual"] / scale, "similarity": round(best[(a, b)], 4),
                      "rank": min(rank[(a, b)], rank[(b, a)])})  # fmt: skip
    return edges, timing


def groups_from_links(names: list[str], edges: list[dict], *, min_inliers: int = 300,
                      min_share: float = 0.2) -> list[dict]:  # fmt: skip
    # Colosseum, 4000 lifted matches per model pair: true links 24-33 % inliers (scales consistent around
    # loops: 0.176 x 0.441 = 0.078 against 0.084 measured), everything else under 15 %.
    """Maximum spanning forest over confident edges -> groups ``{"root", "members": {name: Sim3 to root}}``."""
    good = sorted((e for e in edges if e["inliers"] >= min_inliers and e["inliers"] >= min_share * e["candidates"]),
                  key=lambda e: -e["inliers"])  # fmt: skip
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    tree: dict[str, list[tuple[str, Sim3]]] = {n: [] for n in names}
    for e in good:
        ra, rb = find(e["a"]), find(e["b"])
        if ra == rb:
            continue
        parent[rb] = ra
        tree[e["a"]].append((e["b"], e["sim3"]))  # b -> a
        tree[e["b"]].append((e["a"], invert(e["sim3"])))  # a -> b
    out, seen = [], set()
    for root in names:  # largest first: each group's root is its largest model
        if root in seen:
            continue
        members, stack = {root: (1.0, np.eye(3), np.zeros(3))}, [root]
        seen.add(root)
        while stack:
            cur = stack.pop()
            for nxt, t_cur_from_nxt in tree[cur]:
                if nxt in seen:
                    continue
                members[nxt] = compose(members[cur], t_cur_from_nxt)
                seen.add(nxt)
                stack.append(nxt)
        out.append({"root": root, "members": members})
    return out


def write_merged(recs: dict[str, Any], members: dict[str, Sim3], out_dir: Path) -> None:
    """The members, each moved into the root's frame, as one COLMAP text model with renumbered ids."""
    import tempfile

    import pycolmap

    cams, rigs, frames, images, points = [], [], [], [], []
    off = {"cam": 0, "rig": 0, "frame": 0, "image": 0, "point": 0}
    for name, (s, r, t) in members.items():
        rec = pycolmap.Reconstruction(recs[name])  # a copy: the caller's model is not changed
        rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(r), t))
        with tempfile.TemporaryDirectory() as tmp:
            rec.write_text(tmp)
            txt = {f: [line.split() for line in (Path(tmp) / f).read_text().splitlines() if line and not line.startswith("#")]
                   for f in ("cameras.txt", "rigs.txt", "frames.txt", "points3D.txt")}  # fmt: skip
            # images.txt comes in line pairs, and an image without points has an empty second line
            txt["images.txt"] = [line.split() for line in (Path(tmp) / "images.txt").read_text().splitlines() if not line.startswith("#")]
        cmap = {int(c[0]): off["cam"] + k + 1 for k, c in enumerate(txt["cameras.txt"])}
        rmap = {int(c[0]): off["rig"] + k + 1 for k, c in enumerate(txt["rigs.txt"])}
        fmap = {int(c[0]): off["frame"] + k + 1 for k, c in enumerate(txt["frames.txt"])}
        img_rows = txt["images.txt"]
        imap = {int(img_rows[k][0]): off["image"] + k // 2 + 1 for k in range(0, len(img_rows), 2)}
        pmap = {int(p[0]): off["point"] + k + 1 for k, p in enumerate(txt["points3D.txt"])}
        for c in txt["cameras.txt"]:
            cams.append([str(cmap[int(c[0])]), *c[1:]])
        for rg in txt["rigs.txt"]:  # RIG_ID NUM_SENSORS REF_TYPE REF_ID [TYPE ID ...]
            rigs.append([str(rmap[int(rg[0])]), rg[1], rg[2], str(cmap[int(rg[3])]), *rg[4:]])
        for fr in txt["frames.txt"]:  # FRAME_ID RIG_ID QW QX QY QZ TX TY TZ NUM_DATA (TYPE SENSOR_ID DATA_ID)*
            data = fr[10:]
            for k in range(0, len(data), 3):
                data[k + 1], data[k + 2] = str(cmap[int(data[k + 1])]), str(imap[int(data[k + 2])])
            frames.append([str(fmap[int(fr[0])]), str(rmap[int(fr[1])]), *fr[2:10], *data])
        for k in range(0, len(img_rows), 2):  # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME / (X Y POINT3D_ID)*
            head, pts = img_rows[k], img_rows[k + 1] if k + 1 < len(img_rows) else []
            pts = [pts[j] if (j % 3) != 2 else (str(pmap[int(pts[j])]) if int(pts[j]) >= 0 else "-1") for j in range(len(pts))]
            images.append([str(imap[int(head[0])]), *head[1:8], str(cmap[int(head[8])]), *head[9:]])
            images.append(pts)
        for p in txt["points3D.txt"]:  # POINT3D_ID X Y Z R G B ERROR (IMAGE_ID POINT2D_IDX)*
            track = p[8:]
            track = [str(imap[int(track[j])]) if j % 2 == 0 else track[j] for j in range(len(track))]
            points.append([str(pmap[int(p[0])]), *p[1:8], *track])
        off = {"cam": off["cam"] + len(cmap), "rig": off["rig"] + len(rmap), "frame": off["frame"] + len(fmap),
               "image": off["image"] + len(imap), "point": off["point"] + len(pmap)}  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    for f, rows in (("cameras.txt", cams), ("rigs.txt", rigs), ("frames.txt", frames), ("images.txt", images), ("points3D.txt", points)):
        (out_dir / f).write_text("".join(" ".join(r) + "\n" for r in rows))
    merged = pycolmap.Reconstruction(str(out_dir))  # read back: the ids and tracks are consistent
    merged.write_binary(str(out_dir))
    for f in ("cameras.txt", "rigs.txt", "frames.txt", "images.txt", "points3D.txt"):
        (out_dir / f).unlink()


def merge_models(model_dirs: list[Path], images: Path, out_root: Path, **kw: Any) -> dict:
    """Merge what links; -> ``{"groups": [{"path", "members", "images"}], "edges", "timing"}`` (largest first)."""
    import pycolmap

    recs = {p.name: pycolmap.Reconstruction(str(p)) for p in model_dirs}
    names = sorted(recs, key=lambda k: -recs[k].num_reg_images())
    edges, timing = find_links(recs, images, **{k: v for k, v in kw.items() if k in ("pairs_per_model_pair", "setting")})
    groups = groups_from_links(names, edges, **{k: v for k, v in kw.items() if k in ("min_inliers", "min_share")})
    t0 = time.perf_counter()
    out = []
    for k, g in enumerate(sorted(groups, key=lambda g: -sum(recs[n].num_reg_images() for n in g["members"]))):
        dst = out_root / str(k)
        write_merged(recs, g["members"], dst)
        out.append({"path": str(dst), "members": sorted(g["members"]), "images": sum(recs[n].num_reg_images() for n in g["members"])})
    timing["write"] = time.perf_counter() - t0
    log.info("merge: %d models -> %d (%s)", len(names), len(out), ", ".join(f"{len(g['members'])}:{g['images']}" for g in out))
    return {"groups": out, "edges": [{k: v for k, v in e.items() if k != "sim3"} | {"scale": round(e["sim3"][0], 4)} for e in edges],
            "timing": {k: round(v, 2) for k, v in timing.items()}}  # fmt: skip
