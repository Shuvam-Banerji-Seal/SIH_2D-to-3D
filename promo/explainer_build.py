"""Build, render and composite the explainer film ("How it works"), every example from our own runs.

    uv run python promo/explainer_assets.py        # the real footage (promo/build/explainer/*.mp4)
    uv run python promo/explainer_build.py [--no-render | --anim-only | --compose-only]

1. FACTS from the runs' JSON (the sources are listed in ``build/explainer/facts.json``; a missing one
   shows a dash, never an invented number) and the soundtrack engine go into ``explainer.html``.
2. The javascript-animation skill's ``render.mjs`` renders it, with its synthesized score.
3. Each slot in ``window.SLOTS`` gets its footage (held on its last frame if shorter), overlaid with
   ffmpeg: ``build/explainer/explainer.mp4``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from drone3d.io.nvdec import ffmpeg_bin  # noqa: E402

SKILL = ROOT / "third_party" / "javascript-animation-skills" / "skills"
BUILD = ROOT / "promo" / "build" / "explainer"
OUT = ROOT / "outputs"
FF = ffmpeg_bin()
HR, COL = OUT / "new_highrise_orbit", OUT / "merge_colosseum"


def load(p: Path):  # type: ignore[no-untyped-def]
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def facts() -> tuple[dict, dict]:
    from names import video_name

    f, src = {}, {}
    ing = load(HR / "ingest" / "result.json")
    if ing:
        f["hr_video_s"] = round(ing["video"]["duration_s"])
        src["hr_video_s"] = str(HR / "ingest/result.json")
    kf = load(HR / "keyframes" / "result.json")
    if kf:
        f["hr_keyframes"] = kf["num_keyframes"]
        src["hr_keyframes"] = str(HR / "keyframes/result.json")
    sfm = load(HR / "sfm" / "result.json")
    if sfm:
        f["hr_registered"] = f"{sfm['registered_images']}/{sfm['input_images']}"
        src["hr_registered"] = str(HR / "sfm/result.json")
    sp = load(HR / "export" / "complete" / "splats_360.json")
    if sp:
        f["hr_flown_deg"] = round(sp["headings_flown_deg"])
        f["hr_views"] = sp["views"]
        n = sp.get("num_splats")
        f["hr_splats"] = f"{n / 1e6:.1f} million" if n and n >= 1e6 else (f"{n:,}" if n else None)
        src["hr_flown_deg"] = str(HR / "export/complete/splats_360.json")
    comp = load(HR / "export" / "complete" / "result.json")
    if comp:
        f["hr_photo_share"] = f"{100 * (1 - comp['generated_share']):.0f} %"
        f["hr_triangles"] = f"{comp['subject_triangles'] / 1e3:.0f}k"
        src["hr_photo_share"] = str(HR / "export/complete/result.json")
    csfm = load(COL / "sfm" / "result.json")
    if csfm and (csfm.get("merge") or {}).get("status") == "ok":
        f["col_before"], f["col_after"] = csfm["merge"]["models_before"], csfm["merge"]["models_after"]
        src["col_before"] = str(COL / "sfm/result.json") + " (merge)"
    cexp = load(COL / "export" / "result.json")
    if cexp:
        f["col_texture_views"] = max((m.get("texture") or {}).get("views") or 0 for m in cexp["models"]) or None
        src["col_texture_views"] = str(COL / "export/result.json")
    bench = load(ROOT / "paper" / "figures" / "all_maps.json")
    if bench:
        f["bench_within"] = f"{sum(bool(r['fast_within_budget']) for r in bench)} of {len(bench)}"
        f["bench_rows"] = [{"name": video_name(Path(r["video"]).name)[:26], "fast": round(r["fast_s"], 1), "budget": round(r["budget_s"], 1)}
                           for r in sorted(bench, key=lambda r: r["budget_s"])]  # fmt: skip
        src["bench_within"] = "paper/figures/all_maps.json (all 15 sample videos; mesh + point cloud vs 1.5 x the video length)"
    gpu = load(ROOT / "paper" / "figures" / "bench_gpu.json")
    fps = ((gpu or {}).get("decode", {}).get("ffmpeg9_nvdec") or {}).get("source_fps")
    if fps:
        f["decode_fps"] = round(fps)
        src["decode_fps"] = "paper/figures/bench_gpu.json"
    return f, src


def run(cmd: list, **kw) -> None:  # type: ignore[no-untyped-def]
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def slots(page: Path) -> list[dict]:
    js = ("const { chromium } = require('playwright-core');"
          "(async () => { let b; try { b = await chromium.launch({ channel: 'chrome' }); } catch { b = await chromium.launch(); }"
          f"const p = await b.newPage(); await p.goto('file://{page}?render');"
          "console.log(JSON.stringify(await p.evaluate(() => window.SLOTS))); await b.close(); })();")  # fmt: skip
    out = subprocess.run(["node", "-e", js], cwd=ROOT / "promo", capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def fit(src: Path, out: Path, w: int, h: int, dur: float, *, start: float = 0.0, image: bool = False) -> Path:
    """``src`` covering a w x h slot for ``dur`` s; a clip shorter than that holds its last frame."""
    if image:
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x1b1f24,fps=30,format=yuv420p"
        run([FF, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{dur}", "-i", src, "-vf", vf,
             "-c:v", "libx264", "-crf", "16", out])  # fmt: skip
        return out
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},fps=30,"
          f"tpad=stop_mode=clone:stop_duration={dur:.2f},format=yuv420p")  # fmt: skip
    run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start}", "-i", src, "-vf", vf, "-t", f"{dur}",
         "-an", "-c:v", "libx264", "-crf", "16", out])  # fmt: skip
    return out


SOURCES = {  # slot id -> (file, start s, image?)
    "source": (ROOT / "uploads" / "highrise_orbit.webm", 3.0, False),
    "timeline": (HR / "keyframes" / "keyframe_timeline.png", 0.0, True),
    "merge": (ROOT / "promo" / "build" / "merge.mp4", 0.0, False),
    "depth": (ROOT / "promo" / "build" / "depth_tiles.mp4", 0.0, False),
    "clean": (BUILD / "clean_wipe.mp4", 0.0, False),
    "texture": (BUILD / "fly_colosseum.mp4", 0.0, False),
    "complete": (BUILD / "turn_complete.mp4", 0.0, False),
    "splats": (BUILD / "turn_splats.mp4", 0.0, False),
    "console": (BUILD / "console_complete.png", 0.0, True),
    "show_notre_dame": (BUILD / "fly_notre_dame.mp4", 0.0, False),
    "show_reichstag": (BUILD / "fly_reichstag.mp4", 0.0, False),
    "show_rural": (BUILD / "fly_rural.mp4", 0.0, False),
    "show_complete": (BUILD / "turn_complete.mp4", 5.0, False),
}


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    f, src = facts()
    page = (ROOT / "promo" / "explainer.template.html").read_text()
    page = page.replace("/*FACTS*/{}/*END*/", json.dumps(f)).replace("/*GROOVE*/", (SKILL / "soundtrack" / "templates" / "groove.js").read_text())
    html = BUILD / "explainer.html"
    html.write_text(page)
    (BUILD / "facts.json").write_text(json.dumps({"facts": f, "facts_sources": src}, indent=1))
    print(json.dumps({k: v for k, v in f.items() if k != "bench_rows"}, indent=1))
    if "--no-render" in sys.argv:
        return
    anim = BUILD / "explainer_anim.mp4"
    env = {**subprocess.os.environ, "PATH": f"{Path(FF).parent}:" + subprocess.os.environ.get("PATH", "")}
    if "--compose-only" not in sys.argv or not anim.is_file():
        run(["node", SKILL / "javascript-animation" / "scripts" / "render.mjs", html, anim], cwd=ROOT / "promo", env=env)
    if "--anim-only" in sys.argv:
        return
    made = {}
    for s in slots(html):
        file, start, image = SOURCES.get(s["id"], (None, 0, False))
        if file is None or not Path(file).is_file():
            print("no source for slot", s["id"])
            continue
        made[s["id"]] = fit(Path(file), BUILD / f"slot_{s['id']}.mp4", s["w"], s["h"], s["t1"] - s["t0"], start=start, image=image)
    inputs, chain, last = ["-i", str(anim)], [], "[0:v]"
    for k, s in enumerate(s for s in slots(html) if s["id"] in made):
        inputs += ["-itsoffset", f"{s['t0']:.3f}", "-i", str(made[s["id"]])]
        full = s["w"] == 1920
        fade = "" if full else f",fade=t=in:st={s['t0']:.3f}:d=0.35:alpha=1,fade=t=out:st={s['t1'] - 0.3:.3f}:d=0.3:alpha=1"
        chain.append(f"[{k + 1}:v]format=yuva420p{fade}[s{k}]")
        chain.append(f"{last}[s{k}]overlay={s['x']}:{s['y']}:enable='between(t,{s['t0']:.3f},{s['t1']:.3f})'[v{k}]")
        last = f"[v{k}]"
    out = BUILD / "explainer.mp4"
    run([FF, "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain), "-map", last, "-map", "0:a?",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", out])  # fmt: skip
    print("wrote", out, "slots:", sorted(made))


if __name__ == "__main__":
    main()
