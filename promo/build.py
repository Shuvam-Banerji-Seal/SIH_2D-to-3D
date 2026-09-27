"""Build the promo film page: fill FACTS from run JSON, inline the soundtrack engine.

    uv run python promo/build.py [RUN_DIR]      # default outputs/jal_mahal

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
        psnr = best.get("eval_metrics", {}).get("psnr")
        f["psnr"] = round(psnr, 1) if psnr else None
        src["psnr"] = str(run / "splat/result.json") + " (held-out keyframes)"
    geo = load(ROOT / "paper" / "figures" / "georef_study.json")
    if geo:
        f["scale_err"] = f"{100 * max(r['scale_rel_err'] for r in geo):.1f} %"
        src["scale_err"] = "paper/figures/georef_study.json (simulation)"
    bench = load(ROOT / "paper" / "figures" / "bench_gpu.json")
    fps = ((bench or {}).get("decode", {}).get("ffmpeg9_nvdec") or {}).get("source_fps")
    if fps:
        f["decode_fps"] = round(fps)
        src["decode_fps"] = "paper/figures/bench_gpu.json"
    return f, src


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "jal_mahal"
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
