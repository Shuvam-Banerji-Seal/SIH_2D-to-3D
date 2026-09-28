"""Build the promo film page: fill FACTS from run JSON, inline the soundtrack engine.

    uv run python promo/build.py [RUN_DIR]      # default outputs/merge_colosseum

Every number the film shows comes from a file listed in ``facts_sources`` in the
output ``promo/build/facts.json``; a missing source leaves the film showing a
dash rather than a made-up value.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "third_party" / "javascript-animation-skills" / "skills"
OUT = ROOT / "promo" / "build"


def load(path: Path):  # type: ignore[no-untyped-def]
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def facts(run: Path) -> tuple[dict, dict]:
    f: dict = {}
    src: dict = {}
    kf = load(run / "keyframes" / "result.json")
    if kf:
        f["passes"] = len(kf["passes"])
        f["keyframes"] = kf["num_keyframes"]
        views = [p["views_per_point"] for p in kf["passes"] if p.get("views_per_point")]
        f["views"] = round(statistics.median(views)) if views else None
        snrs = [p["parallax_snr"] for p in kf["passes"] if p.get("parallax_snr") is not None]
        f["snr_real"] = (
            round(statistics.median(snrs), 1) if snrs else None
        )  # the median pass, not the best
        src.update(passes=str(run / "keyframes/result.json"))
    control = load(
        ROOT / "outputs" / "controls" / "pure_rotation_run" / "keyframes" / "result.json"
    )
    if control and control.get("passes"):
        f["snr_control"] = round(control["passes"][0]["parallax_snr"], 1)
        src["snr_control"] = "outputs/controls/pure_rotation_run/keyframes/result.json"
    sfm = load(run / "sfm" / "result.json")
    if sfm:
        f["registered"] = f"{sfm['registered_images']}/{sfm['input_images']}"
        f["sfm_points"] = f"{sum(m['points'] or 0 for m in sfm['models']):,}"
        src["registered"] = str(run / "sfm/result.json")
    depth = load(run / "depth" / "result.json")
    if depth and depth.get("per_model"):
        largest = max(depth["per_model"].values(), key=lambda v: v["images"])
        v = largest.get("cv_monotone_abs_rel_median")
        f["depth_absrel"] = f"{100 * v:.1f} %" if v is not None else None
        src["depth_absrel"] = (
            str(run / "depth/result.json") + " (largest model, held-out tie points)"
        )
    splat = load(run / "splat" / "result.json")
    if splat and splat.get("models"):
        best = max(splat["models"], key=lambda m: m.get("num_splats") or 0)
        f["splats"] = f"{best['num_splats']:,}" if best.get("num_splats") else None
        ev = best.get("eval_metrics") or {}
        psnr = ev.get("avg_psnr", ev.get("psnr"))
        if isinstance(psnr, list):  # per held-out view
            psnr = sum(psnr) / len(psnr) if psnr else None
        f["psnr"] = round(psnr, 1) if isinstance(psnr, (int, float)) else None
        src["psnr"] = str(run / "splat/result.json") + " (held-out keyframes)"
    geo = load(ROOT / "paper" / "figures" / "georef_study.json")
    if geo:
        f["scale_err"] = f"{100 * max(r['scale_rel_err'] for r in geo):.1f} %"
        src["scale_err"] = "paper/figures/georef_study.json (simulation)"
    # fast profile (the deliverable): poses from flow, fused textured mesh, time vs budget
    ff = load(ROOT / "paper" / "figures" / "fast_flow_sfm.json")
    if ff and ff.get("rows"):
        final = next((r for r in ff["rows"] if r["tracks"].startswith("960") and r["mapper"] == "global"), ff["rows"][-1])
        f["pose_agreement"] = f"{100 * final['centre_rmse_rel_extent']:.1f} %"
        src["pose_agreement"] = "paper/figures/fast_flow_sfm.json (camera centres vs SIFT SfM, share of the flight extent)"
    dense = load(run / "dense" / "result.json")
    if dense and dense.get("models"):
        tri = sum(m.get("mesh_triangles") or 0 for m in dense["models"] if m.get("status") == "ok")
        f["triangles"] = f"{tri / 1e6:.1f} million" if tri >= 1e6 else f"{tri:,}"
        src["triangles"] = str(run / "dense/result.json")
    if sfm and dense:
        sys.path.insert(0, str(ROOT / "src"))
        from drone3d.metrics.quality import scene_view_completeness

        comp = scene_view_completeness(sfm, dense)
        if comp is not None:
            f["completeness"] = f"{100 * comp:.0f} %"
            src["completeness"] = str(run / "dense/result.json") + " (non-sky keyframe pixels the mesh covers, all models)"
    merge = (sfm or {}).get("merge") or {}
    if merge.get("status") == "ok":
        f["merge_before"], f["merge_after"] = merge["models_before"], merge["models_after"]
        largest = max(sfm["models"], key=lambda m: m.get("images") or 0)
        f["merge_largest_shots"] = len(largest.get("passes") or {}) or None
        src["merge_before"] = str(run / "sfm/result.json") + " (merge)"
    anim = load(OUT / "merge.json")  # promo/assets.py: how the merge footage times its passes
    if anim:
        f["merge_anim"] = {k: anim[k] for k in ("shown", "first_s", "stagger_s")}
        src["merge_anim"] = "promo/build/merge.json"
    export = load(run / "export" / "result.json")
    if export and export.get("models"):
        tv = [(m.get("texture") or {}).get("views") for m in export["models"]]
        f["texture_views"] = max((v for v in tv if v), default=None)
        src["texture_views"] = str(run / "export/result.json")
    ing = load(run / "ingest" / "result.json")
    if ing:
        sys.path.insert(0, str(ROOT / "experiments"))
        from names import video_name

        f["site"] = video_name(Path(ing["video"]["path"]).name)
    bench = load(ROOT / "paper" / "figures" / "all_maps.json")
    rows = bench.get("rows") if isinstance(bench, dict) else bench
    if rows:
        f["budget_share"] = f"{sum(bool(r.get('fast_within_budget')) for r in rows)} of {len(rows)}"
        src["budget_share"] = "paper/figures/all_maps.json (the all-videos benchmark, mesh + cloud vs 1.5 x video length)"
    metrics = load(run / "metrics" / "metrics.json") or {}
    proc = metrics.get("processing") or {}
    if proc.get("seconds") and proc.get("video_seconds"):
        f["proc_time"] = f"{proc['seconds']:.0f} s"
        f["video_len"] = f"{proc['video_seconds']:.0f} s"
        src["proc_time"] = str(run / "metrics/metrics.json")
    geo_e2e = load(ROOT / "paper" / "figures" / "fast_georef_e2e.json")
    if geo_e2e:
        near = [((m.get("mesh_error_m") or {}).get("by_distance_from_track") or {}).get("0-100 m", {}).get("median") for m in geo_e2e["models"]]
        near = [x for x in near if x is not None]
        if near:
            f["near_track"] = f"{min(near):.1f}–{max(near):.1f} m"
            src["near_track"] = "paper/figures/fast_georef_e2e.json (synthetic GPS on real models, median within 100 m of the track)"
    bench = load(ROOT / "paper" / "figures" / "bench_gpu.json")
    fps = ((bench or {}).get("decode", {}).get("ffmpeg9_nvdec") or {}).get("source_fps")
    if fps:
        f["decode_fps"] = round(fps)
        src["decode_fps"] = "paper/figures/bench_gpu.json"
    return f, src


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "merge_colosseum"
    OUT.mkdir(parents=True, exist_ok=True)
    f, src = facts(run)
    page = (ROOT / "promo" / "film.template.html").read_text()
    page = page.replace("/*FACTS*/{}/*END*/", json.dumps(f))
    page = page.replace(
        "/*GROOVE*/", (SKILL / "soundtrack" / "templates" / "groove.js").read_text()
    )
    (OUT / "film.html").write_text(page)
    (OUT / "facts.json").write_text(json.dumps({"facts": f, "facts_sources": src}, indent=1))
    print(json.dumps(f, indent=1))


if __name__ == "__main__":
    main()
