"""Render the promo animation and composite real pipeline output into its slots.

    uv run python promo/compose.py [RUN_DIR]        # default outputs/jal_mahal

1. ``promo/build.py`` fills the page's FACTS; the javascript-animation skill's
   ``render.mjs`` renders it (with its synthesized score) to ``build/film_anim.mp4``.
2. Each slot the page declares in ``window.SLOTS`` gets a real source, prepared
   at the slot's size: the source footage, the keyframe timeline figure, a
   keyframe | depth strip made from the run's Marigold output, and the splat
   fly-through.
3. ffmpeg overlays every source inside its slot's time window (fading in and
   out) and keeps the animation's audio. Output: ``promo/build/promo.mp4``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drone3d.io.nvdec import ffmpeg_bin  # noqa: E402

BUILD = ROOT / "promo" / "build"
SKILL = (
    ROOT
    / "third_party"
    / "javascript-animation-skills"
    / "skills"
    / "javascript-animation"
    / "scripts"
)
FF = ffmpeg_bin()


def run(cmd: list[str], **kw) -> None:  # type: ignore[no-untyped-def]
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def slots(page: Path) -> list[dict]:
    """Evaluate window.SLOTS by loading the page in headless Chrome (via node)."""
    js = (
        "const { chromium } = require('playwright-core');"
        "(async () => { let b; try { b = await chromium.launch({ channel: 'chrome' }); } catch { b = await chromium.launch(); }"
        f"const p = await b.newPage(); await p.goto('file://{page}?render');"
        "console.log(JSON.stringify(await p.evaluate(() => window.SLOTS))); await b.close(); })();"
    )
    out = subprocess.run(
        ["node", "-e", js], cwd=ROOT / "promo", capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def fit_video(
    src: Path,
    out: Path,
    w: int,
    h: int,
    start: float,
    dur: float,
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    vf = []
    if crop:
        x0, y0, x1, y1 = crop
        vf.append(f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}")
    vf += [
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={w}:{h}",
        "fps=30",
        "format=yuv420p",
    ]
    run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", src,
         "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-crf", "16", out])  # fmt: skip
    return out


def fit_image(src: Path, out: Path, w: int, h: int, dur: float) -> Path:
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x061222,fps=30,format=yuv420p"
    run([FF, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{dur}", "-i", src,
         "-vf", vf, "-c:v", "libx264", "-crf", "16", out])  # fmt: skip
    return out


def depth_strip(run_dir: Path, out: Path, w: int, h: int, dur: float) -> Path | None:
    """Keyframe | calibrated Marigold depth, one keyframe per ~second, from the run's own files."""
    import cv2

    depth = json.loads((run_dir / "depth" / "result.json").read_text())
    names = sorted(n for n, r in depth["per_image"].items() if "abs_rel" in r)
    if not names:
        return None
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, max(2, int(dur)))]
    tiles = BUILD / "depth_tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    half = (w // 2) // 2 * 2
    for k, name in enumerate(picks):
        rgb = cv2.imread(str(run_dir / "dataset" / "images" / name))
        pred = np.load(
            run_dir / "dataset" / "depth_raw" / (str(Path(name).with_suffix("")) + ".npz")
        )["pred"].astype(np.float32)
        lo, hi = np.quantile(pred, [0.02, 0.98])
        vis = cv2.applyColorMap(
            (255 * np.clip((pred - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        left = cv2.resize(rgb, (half, h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(vis, (w - half, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(tiles / f"t{k:03d}.png"), np.hstack([left, right]))
    per = dur / len(picks)
    run([FF, "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{1 / per}", "-i", tiles / "t%03d.png",
         "-vf", "fps=30,format=yuv420p", "-t", f"{dur}", "-c:v", "libx264", "-crf", "16", out])  # fmt: skip
    return out


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "jal_mahal"
    BUILD.mkdir(parents=True, exist_ok=True)
    run([sys.executable, ROOT / "promo" / "build.py", run_dir])
    page = BUILD / "film.html"
    anim = BUILD / "film_anim.mp4"
    env_path = f"{Path(FF).parent}:" + subprocess.os.environ.get("PATH", "")
    run(
        ["node", SKILL / "render.mjs", page, anim],
        cwd=ROOT / "promo",
        env={**subprocess.os.environ, "PATH": env_path},
    )

    ingest = json.loads((run_dir / "ingest" / "result.json").read_text())
    video = Path(ingest["video"]["path"])
    crop = (json.loads((run_dir / "keyframes" / "result.json").read_text()) or {}).get("crop")
    renders = (
        sorted((run_dir / "render").glob("*_flythrough.mp4"))
        if (run_dir / "render").is_dir()
        else []
    )
    fly = max(renders, key=lambda p: p.stat().st_size) if renders else None

    sources = {}
    for s in slots(page):
        dur, w, h, out = s["t1"] - s["t0"], s["w"], s["h"], BUILD / f"slot_{s['id']}.mp4"
        if s["id"] == "source":
            sources[s["id"]] = fit_video(video, out, w, h, 18.0, dur, tuple(crop) if crop else None)
        elif s["id"] == "timeline":
            sources[s["id"]] = fit_image(
                run_dir / "keyframes" / "keyframe_timeline.png", out, w, h, dur
            )
        elif s["id"] == "depth" and (BUILD / "depth_tiles.mp4").is_file():  # fast profile (promo/assets.py)
            sources[s["id"]] = fit_video(BUILD / "depth_tiles.mp4", out, w, h, 0.0, dur)
        elif s["id"] == "depth":
            made = depth_strip(run_dir, out, w, h, dur)
            if made:
                sources[s["id"]] = made
        elif s["id"] == "splat_inset" and (BUILD / "model_inset.mp4").is_file():
            sources[s["id"]] = fit_video(BUILD / "model_inset.mp4", out, w, h, 0.0, dur)
        elif s["id"] == "flythrough" and (BUILD / "flythrough.mp4").is_file():
            sources[s["id"]] = fit_video(BUILD / "flythrough.mp4", out, w, h, 0.0, dur)
        elif s["id"] in ("splat_inset", "flythrough") and fly:
            sources[s["id"]] = fit_video(
                fly, out, w, h, 0.0 if s["id"] == "splat_inset" else 1.0, dur
            )
    missing = [s["id"] for s in slots(page) if s["id"] not in sources]
    if missing:
        print(f"no source for slot(s) {missing}; they stay as drawn")

    inputs = ["-i", str(anim)]
    chain, last = [], "[0:v]"
    for k, s in enumerate(s for s in slots(page) if s["id"] in sources):
        inputs += ["-itsoffset", f"{s['t0']:.3f}", "-i", str(sources[s["id"]])]
        d = s["t1"] - s["t0"]
        chain.append(f"[{k + 1}:v]format=yuva420p,fade=t=in:st={s['t0']:.3f}:d=0.35:alpha=1,"
                     f"fade=t=out:st={s['t1'] - 0.3:.3f}:d=0.3:alpha=1[s{k}]")  # fmt: skip
        chain.append(
            f"{last}[s{k}]overlay={s['x']}:{s['y']}:enable='between(t,{s['t0']:.3f},{s['t0'] + d:.3f})'[v{k}]"
        )
        last = f"[v{k}]"
    out = BUILD / "promo.mp4"
    run([FF, "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain) or "null",
         "-map", last if chain else "0:v", "-map", "0:a?", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out])  # fmt: skip
    info = subprocess.run([FF.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                          capture_output=True, text=True).stdout.strip()  # fmt: skip
    print(f"wrote {out} ({float(info):.1f} s), slots filled: {sorted(sources)}")


if __name__ == "__main__":
    main()
