"""Do the merge and the subject focus hold on every sample video, not just the ones they were built on?

    uv run python experiments/generality.py [RUN ...]   (default: every outputs/map_* run; writes paper/figures/generality.json)

For each run's per-pass models (the benchmark's map_* runs were mapped before merging):

- merge: the groups RoMa v2 + Sim(3) would form, and for every merged member how far its own fused
  surface lies from the other members' after the similarity -- the median gap where they overlap, as a
  share of the group's extent (a false link shows as a large gap or no overlap);
- subject: per model, whether the optical axes converge on a subject (export.stage._subject), the share of
  keyframes aimed within 25 degrees of it, and the share of the mesh the viewer's focus box keeps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)

ROOT = Path(__file__).resolve().parents[1]


def _mesh_points(run: Path, k: str, n: int = 30000) -> np.ndarray | None:
    import open3d as o3d

    f = run / "dense" / f"model_{k}" / "mesh.ply"
    if not f.is_file():
        return None
    v = np.asarray(o3d.io.read_triangle_mesh(str(f)).vertices)
    if not len(v):
        return None
    return v[np.random.default_rng(0).choice(len(v), min(n, len(v)), replace=False)]


def _poses(rec):  # type: ignore[no-untyped-def]
    ims = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if im.has_pose]
    cams = np.array([im.projection_center() for im in ims])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in ims])
    rots = np.array([im.cam_from_world().rotation.matrix() for im in ims])
    return cams, axes, rots


def subject_row(run: Path, k: str, rec) -> dict:  # type: ignore[no-untyped-def]
    from drone3d.export.stage import _level, _subject

    cams, axes, rots = _poses(rec)
    row: dict = {"model": k, "keyframes": len(cams)}
    sub = _subject(cams, axes)
    if sub is None:
        return {**row, "subject": False}
    point, radius = sub
    rel = point - cams
    cos = np.einsum("ij,ij->i", rel, axes) / np.maximum(np.linalg.norm(rel, axis=1), 1e-12)
    row.update(subject=True, aimed_share=round(float((cos > np.cos(np.radians(25))).mean()), 3),
               in_front_share=round(float((cos > 0).mean()), 3))  # fmt: skip
    pts = _mesh_points(run, k)
    if pts is not None:
        rot = _level(pts, cams, rots)
        p, c = pts @ rot.T, point @ rot.T
        inside = (np.abs(p[:, 0] - c[0]) <= radius) & (np.abs(p[:, 1] - c[1]) <= radius)
        row["mesh_in_box"] = round(float(inside.mean()), 3)
    return row


def merge_rows(run: Path, recs: dict) -> dict:  # type: ignore[no-untyped-def]
    from scipy.spatial import cKDTree

    from drone3d.fastsfm.merge import find_links, groups_from_links

    names = sorted(recs, key=lambda k: -recs[k].num_reg_images())
    edges, timing = find_links(recs, run / "dataset" / "images", max_partners=999)  # every pair: the ranks of true links
    groups = groups_from_links(names, edges)
    out = {"models": len(names), "groups": [], "timing": {k: round(v, 1) for k, v in timing.items()},
           "links": [{"a": e["a"], "b": e["b"], "inliers": e["inliers"], "share": round(e["inliers"] / e["candidates"], 3),
                      "rank": e["rank"]} for e in edges if e["inliers"] >= 300 and e["inliers"] >= 0.2 * e["candidates"]],
           "rejected_best_share": round(max((e["inliers"] / e["candidates"] for e in edges
                                             if not (e["inliers"] >= 300 and e["inliers"] >= 0.2 * e["candidates"])), default=0.0), 3)}  # fmt: skip
    for g in groups:
        members = g["members"]
        if len(members) < 2:
            continue
        clouds = {}
        for m, (s, r, t) in members.items():
            p = _mesh_points(run, m, 20000)
            if p is not None:
                clouds[m] = s * p @ r.T + t
        allp = np.concatenate(list(clouds.values())) if clouds else np.zeros((0, 3))
        ext = float(np.linalg.norm(np.percentile(allp, 95, 0) - np.percentile(allp, 5, 0))) if len(allp) else 1.0
        gaps = {}
        for m, p in clouds.items():
            others = [q for n, q in clouds.items() if n != m]
            if not others:
                continue
            d, _ = cKDTree(np.concatenate(others)).query(p)
            near = d < 0.05 * ext
            gaps[m] = {"overlap": round(float(near.mean()), 3), "gap": round(float(np.median(d[near]) / ext), 4) if near.any() else None}
        out["groups"].append({"members": sorted(members), "images": sum(recs[m].num_reg_images() for m in members), "gaps": gaps})
    return out


def main() -> None:
    import pycolmap

    runs = [ROOT / "outputs" / r for r in sys.argv[1:]] or sorted((ROOT / "outputs").glob("map_*"))
    dst = ROOT / "paper" / "figures" / "generality.json"
    rows = {r["run"]: r for r in (json.loads(dst.read_text()) if dst.is_file() else [])}
    for run in runs:
        sparse = run / "dataset" / "sparse"
        dirs = sorted((p for p in sparse.iterdir() if p.is_dir()), key=lambda p: int(p.name)) if sparse.is_dir() else []
        if not dirs:
            continue
        recs = {p.name: pycolmap.Reconstruction(str(p)) for p in dirs}
        recs = {k: r for k, r in recs.items() if r.num_reg_images() >= 3}
        row = {"run": run.name, "subject": [subject_row(run, k, r) for k, r in recs.items()]}
        try:
            row["merge"] = merge_rows(run, recs) if len(recs) > 1 else {"models": len(recs), "groups": [], "links": []}
        except Exception as exc:  # noqa: BLE001 -- record and go on to the next video
            row["merge"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        rows[run.name] = row
        m = row["merge"]
        print(run.name, "| subject in", sum(s.get("subject", False) for s in row["subject"]), "of", len(row["subject"]),
              "models | merge", m.get("models"), "->", m.get("models", 0) - sum(len(g["members"]) - 1 for g in m.get("groups", [])),
              "| worst gap", max((x["gap"] or 1) for g in m.get("groups", []) for x in g["gaps"].values()) if m.get("groups") else None,
              flush=True)  # fmt: skip
        dst.write_text(json.dumps(list(rows.values()), indent=1))


if __name__ == "__main__":
    main()
