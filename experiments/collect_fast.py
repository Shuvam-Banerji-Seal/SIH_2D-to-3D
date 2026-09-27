"""Collect the fast profile's evidence into paper/figures/fast_*.json (committed copies of run outputs).

Each file is copied or assembled from the run or experiment output that produced
it; nothing is typed in. Missing inputs are skipped.

    uv run python experiments/collect_fast.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "outputs"


def load(p: Path) -> dict | list | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    # SfM from flow vs SIFT (Jal Mahal pass 3, 97 keyframes, keypoints at 960 px)
    exp = OUT / "experiments" / "flow_sfm_pass03"
    rows = []
    for tag, label in (("_ls640_it8", "640 px, 8 iterations"), ("_ls640_it12", "640 px, 12 iterations"),
                       ("_final", "960 px, 12 iterations")):  # fmt: skip
        r = load(exp / f"report{tag}.json")
        if not r:
            continue
        for mapper, run in r["runs"].items():
            best = run["models"][0]
            rows.append({"tracks": label, "mapper": mapper, "flow_s": r["tracks"].get("flow_seconds"),
                         "mapping_s": run["timing_s"]["mapping_s"], "registered": best["registered"],
                         "keyframes": r["keyframes"], "reproj_px": best["mean_reprojection_px"],
                         **{k: run.get("vs_reference", {}).get(k) for k in
                            ("centre_rmse_rel_extent", "centre_max_rel_extent", "rotation_err_median_deg", "rotation_err_max_deg")}})  # fmt: skip
    sift = load(OUT / "jal_mahal" / "sfm" / "result.json") or {}
    (FIG / "fast_flow_sfm.json").write_text(json.dumps(
        {"rows": rows, "sift_reference": {"images": sift.get("input_images"), "registered": sift.get("registered_images"),
                                          "seconds": sift.get("seconds"), "reproj_px_4k": sift.get("mean_reprojection_px")}}, indent=1))  # fmt: skip
    for src, dst in ((OUT / "jal_mahal_rel6" / "dense_sweep.json", "fast_tsdf_sweep.json"),
                     (OUT / "experiments" / "georef_e2e_jal_mahal_fast3" / "georef_e2e.json", "fast_georef_e2e.json")):  # fmt: skip
        if src.is_file():
            shutil.copy2(src, FIG / dst)
    timing = {}
    for run in ("jal_mahal_fast3", "qutub_fast2"):
        m = load(OUT / run / "metrics" / "metrics.json") or {}
        k = load(OUT / run / "keyframes" / "result.json") or {}
        timing[run] = {"processing": m.get("processing"), "rate_probe": (k.get("analysis") or {}).get("rate_probe"),
                       "passes": len(k.get("passes", [])), "keyframes": k.get("num_keyframes")}  # fmt: skip
    (FIG / "fast_timing.json").write_text(json.dumps(timing, indent=1))
    # every sample video through the fast profile on the idle GPU (outputs/ded_*)
    comp = load(FIG / "dedicated_completeness.json") or {}
    ded = {}
    for run in sorted(OUT.glob("ded_*")):
        m = load(run / "metrics" / "metrics.json") or {}
        s = load(run / "sfm" / "result.json") or {}
        k = load(run / "keyframes" / "result.json") or {}
        d = load(run / "dense" / "result.json") or {}
        ing = load(run / "ingest" / "result.json") or {}
        if not m.get("processing"):
            continue
        ded[run.name] = {
            "video": Path((ing.get("video") or {}).get("path", run.name)).stem,
            "resolution": [(ing.get("video") or {}).get("width"), (ing.get("video") or {}).get("height")],
            "processing": m["processing"],
            "passes": len(k.get("passes", [])),
            "verdicts": [p["verdict"] for p in k.get("passes", [])],
            "keyframes": s.get("input_images"),
            "registered": s.get("registered_images"),
            "models": len(s.get("models", [])),
            "triangles": sum(x.get("mesh_triangles") or 0 for x in d.get("models", []) if x.get("status") == "ok"),
            "view_completeness": (comp.get(run.name) or {}).get("view_completeness"),
        }
    (FIG / "fast_dedicated_runs.json").write_text(json.dumps(ded, indent=1))
    print("wrote", sorted(p.name for p in FIG.glob("fast_*.json")))


if __name__ == "__main__":
    main()
