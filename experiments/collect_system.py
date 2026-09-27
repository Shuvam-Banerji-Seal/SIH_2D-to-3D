"""Collect the all-videos batch (every sample video, fast profile + splats, warm engine) into paper/figures/all_maps.json.

    uv run python experiments/collect_system.py

Reads outputs/map_*/: per-stage seconds, keyframes, registration, models,
view completeness, splat count and held-out PSNR per model, and the sizes of
the deliverables, so the paper's all-videos table is regenerated from runs.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        fast = sum(v for k, v in stages.items() if k != "splat")
        files = [run / "export" / f for mm in export.get("models", []) for f in mm["files"]]
        rows.append({
            "run": run.name, "video": Path(ing.get("path", "")).name, "seconds": ing.get("duration_s"),
            "resolution": [ing.get("width"), ing.get("height")], "fps": ing.get("fps"),
            "keyframes": sfm.get("input_images"), "registered": sfm.get("registered_images"), "models": len(dense),
            "completeness": round(sum(d["view_completeness"] for d in dense) / len(dense), 4) if dense else None,
            "triangles": sum(d.get("mesh_triangles") or 0 for d in dense),
            "stage_s": stages, "fast_s": round(fast, 1), "splat_s": stages.get("splat"), "total_s": p["seconds"],
            "budget_s": p["budget_seconds"], "fast_within_budget": fast <= p["budget_seconds"],
            "splat_models": len(splat), "splats": sum(s.get("num_splats") or 0 for s in splat),
            "splat_psnr_mean": round(sum(s["psnr"] for s in splat) / len(splat), 2) if splat else None,
            "splat_cc_psnr_mean": round(sum(s["cc_psnr"] for s in splat) / len(splat), 2) if splat else None,
            "deliverable_mb": round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 1),
            "warm": job.get("warm"),
        })  # fmt: skip
    (FIG / "all_maps.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} runs -> {FIG / 'all_maps.json'}")
    for r in rows:
        print(f"  {r['run'][:40]:40} {r['seconds']:6.1f}s kf {r['keyframes']:4} models {r['models']} compl {r['completeness']} "
              f"fast {r['fast_s']:6.1f}/{r['budget_s']:6.1f} splat {r['splat_s']} psnr {r['splat_psnr_mean']}")  # fmt: skip


if __name__ == "__main__":
    main()
