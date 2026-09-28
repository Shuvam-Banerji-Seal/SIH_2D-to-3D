"""Can the per-pass SfM models of one video be put into one frame? RoMa v2 across passes, Sim(3) between models.

    uv run python experiments/merge_passes.py RUN   (writes paper/figures/merge_<RUN>.json)

The fast profile maps each pass (shot) of a video on its own: an edit that circles a building from five sides
gives five models, none of which knows the others. For every pair of models this

1. retrieves the keyframe pairs most likely to overlap -- cosine similarity of mean-pooled DINOv3 features
   (RoMa v2's own backbone, so nothing gated is needed);
2. matches them densely with RoMa v2 and keeps the confident matches;
3. lifts a match to 3D where it lands within a few pixels of an SfM observation in both images (the flow
   tracks give ~1000 per keyframe), giving 3D-3D correspondences between the two models;
4. fits a similarity (Umeyama in RANSAC) and reports its inliers and residual.

Models linked by a confident similarity form one merged model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)

ROOT = Path(__file__).resolve().parents[1]


def umeyama(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |s R a + t - b|."""
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = a - ma, b - mb
    u, d, vt = np.linalg.svd(cb.T @ ca / len(a))
    s_ = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_[2, 2] = -1
    r = u @ s_ @ vt
    s = float(np.trace(np.diag(d) @ s_) / ca.var(0).sum())
    return s, r, mb - s * r @ ma


def ransac_sim3(a: np.ndarray, b: np.ndarray, thresh: float, iters: int = 2000, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(iters):
        idx = rng.choice(len(a), 4, replace=False)
        s, r, t = umeyama(a[idx], b[idx])
        if not np.isfinite(s) or s <= 0:
            continue
        res = np.linalg.norm(s * a @ r.T + t - b, axis=1)
        n = int((res < thresh).sum())
        if n > best[0]:
            best = (n, res < thresh)
    if best[1] is None or best[0] < 8:
        return None
    inl = best[1]
    s, r, t = umeyama(a[inl], b[inl])
    res = np.linalg.norm(s * a @ r.T + t - b, axis=1)
    inl = res < thresh
    return {"scale": s, "rotation": r, "translation": t, "inliers": int(inl.sum()), "candidates": len(a),
            "median_residual": float(np.median(res[inl]))}  # fmt: skip


def main() -> None:
    import pycolmap
    import torch
    from PIL import Image
    from romav2 import RoMaV2

    run = ROOT / "outputs" / sys.argv[1]
    images = run / "dataset" / "images"
    sfm = json.loads((run / "sfm" / "result.json").read_text())
    recs = {Path(m["path"]).name: pycolmap.Reconstruction(m["path"]) for m in sfm["models"]}
    t0 = time.perf_counter()
    cfg = RoMaV2.Cfg(setting="fast")
    roma = RoMaV2(cfg).cuda().eval()
    t_load = time.perf_counter() - t0

    # 1. a global descriptor per registered keyframe, from RoMa's DINOv3 backbone
    keys, desc = [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for k, rec in recs.items():
            for im in rec.images.values():
                if not im.has_pose:
                    continue
                x = torch.from_numpy(np.asarray(Image.open(images / im.name).convert("RGB").resize((448, 256)))).cuda()
                feats = roma.f(x.permute(2, 0, 1)[None].float() / 255)[-1]  # [1, h, w, D] of the deeper layer
                keys.append((k, im.image_id, im.name))
                desc.append(torch.nn.functional.normalize(feats.float().mean(dim=(1, 2))[0], dim=0).cpu())
    d = torch.stack(desc).numpy()
    sim = d @ d.T
    t_desc = time.perf_counter() - t0

    # 2-4. per model pair: top keyframe pairs, RoMa, 3D-3D, Sim(3)
    model_of = np.array([k for k, _, _ in keys])
    names = sorted(recs, key=lambda k: -recs[k].num_reg_images())
    edges, t_match, n_pairs = [], 0.0, 0
    for ai, a in enumerate(names):
        for b in names[ai + 1:]:
            ia, ib = np.where(model_of == a)[0], np.where(model_of == b)[0]
            block = sim[np.ix_(ia, ib)]
            order = np.dstack(np.unravel_index(np.argsort(-block, axis=None), block.shape))[0][:4]
            pa, pb = [], []
            for r_, c_ in order:
                ka, kb = keys[ia[r_]], keys[ib[c_]]
                t1 = time.perf_counter()
                preds = roma.match(str(images / ka[2]), str(images / kb[2]))
                matches, certainty, *_ = roma.sample(preds, 3000)
                t_match += time.perf_counter() - t1
                n_pairs += 1
                ima, imb = recs[a].images[ka[1]], recs[b].images[kb[1]]
                ca, cb = recs[a].cameras[ima.camera_id], recs[b].cameras[imb.camera_id]
                m = matches.float().cpu().numpy()
                xa = np.stack([(m[:, 0] + 1) / 2 * ca.width, (m[:, 1] + 1) / 2 * ca.height], 1)
                xb = np.stack([(m[:, 2] + 1) / 2 * cb.width, (m[:, 3] + 1) / 2 * cb.height], 1)

                def obs(rec, im):  # type: ignore[no-untyped-def]
                    pts = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in im.points2D if p.has_point3D()]
                    return (np.array([q[0] for q in pts]), np.array([q[1] for q in pts])) if pts else (np.zeros((0, 2)), np.zeros((0, 3)))

                oa_xy, oa_X = obs(recs[a], ima)
                ob_xy, ob_X = obs(recs[b], imb)
                if not len(oa_xy) or not len(ob_xy):
                    continue
                from scipy.spatial import cKDTree

                da, ja = cKDTree(oa_xy).query(xa)
                db, jb = cKDTree(ob_xy).query(xb)
                ok = (da < 6) & (db < 6)
                pa.append(oa_X[ja[ok]])
                pb.append(ob_X[jb[ok]])
            if not pa or sum(len(x) for x in pa) < 12:
                continue
            A, B = np.concatenate(pb), np.concatenate(pa)  # B's points -> A's frame
            scale_a = float(np.median(np.linalg.norm(B - B.mean(0), axis=1))) or 1.0
            fit = ransac_sim3(A, B, thresh=0.03 * scale_a)
            if fit is None:
                continue
            edges.append({"a": a, "b": b, "pairs": len(order), "inliers": fit["inliers"], "candidates": fit["candidates"],
                          "inlier_share": round(fit["inliers"] / fit["candidates"], 3),
                          "median_residual_rel": round(fit["median_residual"] / scale_a, 4), "scale": round(fit["scale"], 4)})  # fmt: skip
            print(edges[-1], flush=True)

    # merge graph: confident edges (>= 30 inliers and >= 25 % of the candidates)
    good = [e for e in edges if e["inliers"] >= 30 and e["inlier_share"] >= 0.25]
    parent = {k: k for k in names}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for e in sorted(good, key=lambda e: -e["inliers"]):
        ra, rb = find(e["a"]), find(e["b"])
        if ra != rb:
            parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for k in names:
        groups.setdefault(find(k), []).append(k)
    out = {"run": run.name, "models": {k: recs[k].num_reg_images() for k in names}, "edges": edges,
           "groups": [{"models": g, "images": sum(recs[k].num_reg_images() for k in g)} for g in groups.values()],
           "seconds": {"load": round(t_load, 1), "descriptors": round(t_desc, 1), "matching": round(t_match, 1), "pairs": n_pairs}}  # fmt: skip
    print(json.dumps({k: v for k, v in out.items() if k != "edges"}, indent=1))
    (ROOT / "paper" / "figures" / f"merge_{run.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
