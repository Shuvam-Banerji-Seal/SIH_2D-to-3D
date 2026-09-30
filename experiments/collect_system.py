"""Collect the all-videos batch (every sample video, fast profile + splats, warm engine) into paper/figures/all_maps.json.

    uv run python experiments/collect_system.py

Reads outputs/map_*/: per-stage seconds, keyframes, registration, models,
view completeness, splat count and held-out PSNR per model, and the sizes of
the deliverables, so the paper's all-videos table is regenerated from runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from drone3d.metrics.quality import scene_view_completeness

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "paper" / "figures"


def load(p: Path) -> dict | list | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def main() -> None:
    rows = []
    for run in sorted(OUT.glob("map_*")):
        m = load(run / "metrics" / "metrics.json") or {}
        job = load(run / "ui_job.json") or {}
        p = m.get("processing")
        if not p or job.get("status") != "done":
            continue
        ing = (load(run / "ingest" / "result.json") or {}).get("video") or {}
        sfm = load(run / "sfm" / "result.json") or {}
        dense = [d for d in m.get("dense", []) if d.get("view_completeness") is not None]
        splat = m.get("splat") or []
        export = load(run / "export" / "result.json") or {}
        stages = p["per_stage_s"]
        fast = sum(v for k, v in stages.items() if k not in ("splat", "depth", "mesh", "render"))
        files = [run / "export" / f for mm in export.get("models", []) for f in mm["files"]]
        # other processes' GPU memory during our GPU stages (an idle 1.7 GB server is always there): > 3 GB means
        # someone else was computing, and the timing is re-measured before it is reported
        others = []
        for st in ("keyframes", "sfm", "dense", "export"):
            g = (load(run / st / "result.json") or {}).get("gpu") or {}
            if g.get("memory_peak_gb") is not None and g.get("own_memory_peak_gb") is not None:
                others.append(g["memory_peak_gb"] - g["own_memory_peak_gb"])
        rows.append({
            "run": run.name, "video": Path(ing.get("path", "")).name, "seconds": ing.get("duration_s"),
            "resolution": [ing.get("width"), ing.get("height")], "fps": ing.get("fps"),
            "keyframes": sfm.get("input_images"), "registered": sfm.get("registered_images"), "models": len(dense),
            # every registered keyframe's non-sky pixels; a model without a mesh counts as uncovered
            "completeness": scene_view_completeness(sfm, load(run / "dense" / "result.json") or {}),
            "triangles": sum(d.get("mesh_triangles") or 0 for d in dense),
            "stage_s": stages, "fast_s": round(fast, 1), "splat_s": stages.get("splat"), "total_s": p["seconds"],
            "budget_s": p["budget_seconds"], "fast_within_budget": fast <= p["budget_seconds"],
            "splat_models": len(splat), "splats": sum(s.get("num_splats") or 0 for s in splat),
            "splat_psnr_mean": round(sum(s["psnr"] for s in splat) / len(splat), 2) if splat else None,
            "splat_cc_psnr_mean": round(sum(s["cc_psnr"] for s in splat) / len(splat), 2) if splat else None,
            "deliverable_mb": round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 1),
            "warm": job.get("warm"), "others_gpu_gb": round(max(others), 1) if others else None,
            "clean": bool(others) and max(others) < 3.0,
        })  # fmt: skip
        # benchmark.py --sequential sampled the other processes' SM use too: a crowd of small jobs passes the memory test
        seq = (load(OUT / ".benchmark_load.json") or {}).get(run.name)
        took = (job["finished"] - job["started"]) if job.get("finished") and job.get("started") else None
        if seq and seq.get("seconds") is not None and took is not None and abs(took - seq["seconds"]) < 1.0:  # this run's
            rows[-1].update(clean=bool(seq["clean"]), others_sm_mean=seq["foreign_sm_mean"])
    (FIG / "all_maps.json").write_text(json.dumps(rows, indent=1))

    # analysis rate / overlap comparisons (same engine, one job at a time)
    ablation = []
    for video, prefix in (("Jal Mahal", "jal_mahal"), ("Qutub Minar", "qutub_minar")):
        base = next((r for r in rows if r["run"].startswith(f"map_{prefix}")), None)
        variants = [("adaptive", "absolute", OUT / base["run"])] if base else []
        variants += [("fixed 12 fps", "absolute", next(OUT.glob(f"abl_{prefix[:10]}*_fixed12"), None)),
                     ("adaptive", "relative", next(OUT.glob(f"abl_{prefix[:10]}*_relative"), None))]  # fmt: skip
        for rate, overlap, run in variants:
            if run is None or (load(run / "ui_job.json") or {}).get("status") != "done":
                continue
            m = load(run / "metrics" / "metrics.json") or {}
            k = load(run / "keyframes" / "result.json") or {}
            s_ = load(run / "sfm" / "result.json") or {}
            stages = {st: v for st, v in m["processing"]["per_stage_s"].items() if st != "splat"}
            an = k.get("analysis", {})
            fps = (load(run / "ingest" / "result.json") or {}).get("video", {}).get("fps")
            label = f"{rate}, {fps / an['stride']:.1f} fps" if rate == "adaptive" and fps and an.get("stride") else rate
            ablation.append({"video": video, "rate": label, "overlap": overlap, "keyframes": s_.get("input_images"),
                             "registered": s_.get("registered_images"), "completeness": scene_view_completeness(s_, load(run / "dense" / "result.json") or {}),
                             "seconds": round(sum(stages.values()), 1), "budget_s": m["processing"]["budget_seconds"]})  # fmt: skip
    (FIG / "fast_ablation.json").write_text(json.dumps(ablation, indent=1))
    print(f"{len(ablation)} ablation rows")
    print(f"{len(rows)} runs -> {FIG / 'all_maps.json'}")
    for r in rows:
        print(f"  {'   ' if r['clean'] else 'BUSY'} {r['run'][:40]:40} {r['seconds']:6.1f}s kf {r['keyframes']:4} models {r['models']} compl {r['completeness']} "
              f"fast {r['fast_s']:6.1f}/{r['budget_s']:6.1f} splat {r['splat_s']} psnr {r['splat_psnr_mean']}")  # fmt: skip


if __name__ == "__main__":
    main()
