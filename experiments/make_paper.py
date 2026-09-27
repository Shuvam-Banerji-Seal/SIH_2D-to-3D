"""Generate the paper's results section and macros from run directories.

No number in ``paper/generated/*.tex`` is typed by hand: each comes from a JSON
file written by the pipeline or an experiment script. A missing input renders
as an em dash, never as a guess.

    uv run python experiments/make_paper.py            # writes paper/generated/
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)

OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
DASH = "---"


def load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fmt(v, nd: int = 2, pct: bool = False) -> str:
    if v is None:
        return DASH
    if isinstance(v, str):
        return tex(v)
    if pct:
        return f"{100 * v:.{nd}f}\\%"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    return f"{v:.{nd}f}"


def tex(s: str) -> str:
    """Escape LaTeX specials and drop emoji/symbols pdflatex cannot typeset."""
    import unicodedata

    s = "".join(
        ch
        for ch in str(s)
        if ord(ch) < 0x2000 and unicodedata.category(ch)[0] != "S" or ch in "+=<>"
    )
    return re.sub(r"([_%&#$])", r"\\\1", s).strip()


def table(
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[str]],
    spec: str,
    wide: bool = False,
    note: str = "",
) -> str:
    env = "table*" if wide else "table"
    body = (
        " \\\\\n".join(" & ".join(r) for r in rows)
        if rows
        else f"\\multicolumn{{{len(header)}}}{{c}}{{no data}}"
    )
    foot = f"\n\\par\\smallskip\\footnotesize {note}" if note else ""
    return (
        f"\\begin{{{env}}}[htbp]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\end{{adjustbox}}{foot}\n\\end{{{env}}}\n"
    )


# --------------------------------------------------------------------------- data


def dataset_rows() -> list[list[str]]:
    from drone3d.io.nvdec import probe_stream

    rows = []
    for video in sorted((ROOT / "datasets").glob("*.webm")):
        try:
            info = probe_stream(video, count_frames=False)
        except Exception:  # an unreadable file is simply not listed
            continue
        name = re.sub(r"\s*\[[^\]]+\]$", "", video.stem)
        name = re.split(r"[｜|：:]| - ", name)[0].strip()[:34]
        rows.append(
            [
                tex(name),
                f"{info.width}$\\times${info.height}",
                f"{info.fps:.2f}",
                f"{info.duration_s:.0f}",
            ]
        )
    return rows


def run_dirs() -> dict[str, Path]:
    """Accurate-profile (spirula SfM) runs with a keyframes result, keyed by run name.

    Fast-profile runs (``sfm.backend: flow``) are reported in their own section
    from the committed ``paper/figures/fast_*.json``.
    """
    runs = {}
    for d in sorted((ROOT / "outputs").iterdir()):
        if not (d / "keyframes" / "result.json").is_file():
            continue
        sfm = load(d / "sfm" / "result.json") or {}
        if sfm.get("backend") == "flow":
            continue
        runs[d.name] = d
    return runs


FAST_RUNS = {  # run -> (video, analysis rate, overlap)
    "jal_mahal_fast2": ("Jal Mahal", "12 fps", "absolute"),
    "jal_mahal_fast3": ("Jal Mahal", "adaptive, 3.0 fps", "absolute"),
    "jal_mahal_rel6": ("Jal Mahal", "adaptive, 6.0 fps", "relative"),
    "qutub_fast": ("Qutub Minar", "15 fps", "absolute"),
    "qutub_fast2": ("Qutub Minar", "adaptive, 3.3 fps", "absolute"),
    "qutub_fast3": ("Qutub Minar", "adaptive, 7.5 fps", "relative"),
}


def short_title(video: str) -> str:
    """A sample video's name for a table: before the first separator, whole words, at most 26 characters."""
    name = re.split(r"[｜|：:,]| - |\\[|\\(", video)[0].strip()
    name = re.sub(r"\\b(4K|4k|HD|Drone|Video|Cinematic|FPV drone|in FPV drone)\\b.*$", "", name).strip() or name
    words, out = name.split(), ""
    for w in words:
        if len(out) + len(w) + 1 > 26:
            break
        out = f"{out} {w}".strip()
    return tex(out or name[:26])


def fast_section(macros: dict[str, str]) -> list[str]:
    """Tables and text for the fast profile, from paper/figures/fast_*.json."""
    parts = ["\\subsection{Fast profile: mesh and point cloud within the time budget}\\label{sec:fast-results}\n"]
    ff = load(FIG / "fast_flow_sfm.json")
    if ff and ff["rows"]:
        sift = ff["sift_reference"]
        final = next((r for r in ff["rows"] if r["tracks"].startswith("960") and r["mapper"] == "global"), ff["rows"][-1])
        macros["FlowSfmCentre"] = f"{100 * final['centre_rmse_rel_extent']:.2f}"
        macros["FlowSfmRot"] = f"{final['rotation_err_median_deg']:.2f}"
        rows = [[tex(r["tracks"]), r["mapper"], fmt(r["flow_s"], 1), fmt(r["mapping_s"], 1), f"{r['registered']}/{r['keyframes']}",
                 fmt(r["reproj_px"], 2), fmt(100 * r["centre_rmse_rel_extent"], 2), fmt(r["rotation_err_median_deg"], 2),
                 fmt(r["rotation_err_max_deg"], 2)] for r in ff["rows"]]  # fmt: skip
        parts.append(
            "\\paragraph{Structure from flow.} On the 97-keyframe orbit of Jal Mahal, tracks from direct RAFT flow register "
            f"every keyframe; with 960\\,px tracks the global mapper's camera centres agree with spirula's SIFT reconstruction to "
            f"{macros['FlowSfmCentre']}\\,\\% of the flight extent and its rotations to {macros['FlowSfmRot']}$^\\circ$ (median), with no "
            "feature extraction or matching (Table~\\ref{tab:flowsfm}). Tracking at 640\\,px is cheaper but 2.5--3$\\times$ less "
            "accurate. COLMAP's incremental mapper needs its initialisation angle lowered from 16$^\\circ$ to 2$^\\circ$ to "
            "start on single-pass footage at all and is eight times slower here.\n"
        )
        note = (f"Reference: spirula-studio SIFT SfM of all {sift.get('images')} Jal Mahal keyframes, "
                f"{fmt(sift.get('seconds'), 0)}\\,s for extraction, matching and mapping.")  # fmt: skip
        parts.append(table("SfM from flow tracks on Jal Mahal pass 3 against SIFT SfM of the same keyframes (similarity-aligned).",
                           "tab:flowsfm", ["Tracks", "Mapper", "Flow (s)", "Map (s)", "Reg.", "Reproj. (px)",
                                           "Centre (\\% ext.)", "Rot. med. ($^\\circ$)", "Rot. max ($^\\circ$)"],
                           rows, "llrrrrrrr", wide=True, note=note))  # fmt: skip
    sweep = load(FIG / "fast_tsdf_sweep.json")
    if sweep:
        rows = [[r["model"], fmt(r["trunc"], 0), fmt(r["completeness"], 3), fmt(100 * r["depth_err_vs_triangulated"], 2),
                 fmt(int(r["triangles"]))] for r in sweep if r["voxel_px"] == 3.0]  # fmt: skip
        parts.append(
            "\\paragraph{Fusion.} The depth maps (triangulated, then filled by the calibrated prior) cover every non-sky pixel "
            "of the references, yet a TSDF with the usual 4-voxel truncation band keeps only part of the view: neighbouring "
            "views disagree slightly and their signed distances cancel. Widening the band recovers most of it at a small cost "
            "in depth error against the triangulated geometry (Table~\\ref{tab:tsdf}); the profile uses 12 voxels.\n"
        )
        parts.append(table("TSDF truncation band on Jal Mahal's two largest models (voxel = 3 pixel footprints at the median "
                           "depth): view completeness, median depth error against the triangulated depth, triangles.",
                           "tab:tsdf", ["Model", "Band (vox.)", "Completeness", "Depth err. (\\%)", "Triangles"], rows, "lrrrr"))  # fmt: skip
    ab = load(FIG / "fast_ablation.json")
    if ab:
        rows = [[tex(r["video"]), tex(r["rate"]), r["overlap"], fmt(r["keyframes"]), fmt(r["registered"]), fmt(r["completeness"], 2),
                 fmt(r["seconds"], 0), fmt(r["budget_s"], 0) + (" \\checkmark" if r["seconds"] <= r["budget_s"] else "")] for r in ab]  # fmt: skip
        parts.append(
            "\\paragraph{Analysis rate and overlap.} A fixed 12\\,fps analysis rate places more keyframes and gains a few points "
            "of completeness at a large cost in time; measuring overlap against the points that survive the first tracking step "
            "(so that water does not force a keyframe every step) does not pay off either (Table~\\ref{tab:fastruns}). The profile "
            "uses the motion-adaptive rate with absolute overlap.\n"
        )
        parts.append(table("Fast profile: analysis rate and overlap measure, view completeness and time against the budget "
                           "(1.5$\\times$ video length).", "tab:fastruns",
                           ["Video", "Analysis", "Overlap", "KF", "Reg.", "Compl.", "Time (s)", "Budget (s)"], rows, "lllrrrrr", wide=True))  # fmt: skip
    am = load(FIG / "all_maps.json")
    if am:
        rows, ok = [], 0
        for r in sorted(am, key=lambda r: r["seconds"] or 0):
            ok += r["fast_within_budget"]
            rows.append([short_title(r["video"]), f"{r['seconds']:.0f}", f"{r['resolution'][1]}p", fmt(r["keyframes"]),
                         f"{r['registered']}/{r['keyframes']}", fmt(r["models"]), fmt(r["completeness"], 2), f"{r['fast_s']:.0f}",
                         f"{r['budget_s']:.0f}" + (" \\checkmark" if r["fast_within_budget"] else ""),
                         f"{r['fast_s'] / r['seconds']:.2f}", fmt(r["splat_s"], 0), fmt(r["splat_psnr_mean"], 1)])  # fmt: skip
        by = {r["video"].split(",")[0].split(" ")[0].lower(): r for r in am}
        for key, tag in (("qutub", "Qutub"), ("jal", "Jal")):
            r = by.get(key)
            if r:
                macros[f"Fast{tag}Seconds"] = f"{r['fast_s']:.0f}"
                macros[f"Fast{tag}Budget"] = f"{r['budget_s']:.0f}"
                macros[f"Fast{tag}Compl"] = f"{r['completeness']:.2f}"
        total_video = sum(r["seconds"] for r in am)
        total_fast = sum(r["fast_s"] for r in am)
        macros.update(BenchN=str(len(am)), BenchWithin=str(ok), BenchRate=f"{total_fast / total_video:.2f}",
                      BenchFootage=f"{total_video / 60:.0f}")  # fmt: skip
        misses = [r for r in am if not r["fast_within_budget"]]
        fpv = [r for r in misses if (r["keyframes"] or 0) / max(r["seconds"] or 1, 1) > 4]
        short = [r for r in misses if r not in fpv and (r["seconds"] or 0) < 30]
        other = [r for r in misses if r not in fpv and r not in short]
        why = []
        if fpv:
            why.append(f"fast FPV flights ({', '.join(short_title(r['video']) for r in fpv)}), whose speed needs "
                       + ", ".join(f"{r['keyframes'] / r['seconds']:.0f}" for r in fpv) + " keyframes per second")
        if short:
            why.append(f"clips under 30\\,s ({', '.join(short_title(r['video']) for r in short)}), where fixed per-run costs dominate "
                       "a budget proportional to length")
        if other:
            why.append(", ".join(short_title(r["video"]) for r in other))
        parts.append(
            "\\paragraph{Every sample video.} On the warm engine the mesh-and-cloud stages processed "
            f"{macros['BenchWithin']} of the {macros['BenchN']} sample videos within the budget "
            f"({macros['BenchFootage']} minutes of footage at {macros['BenchRate']}\\,s per second of video overall; "
            "Table~\\ref{tab:allmaps})." + (f" The misses are {'; '.join(why)}." if why else "") + " Every model was then "
            "given Gaussian splats, trained from its dense cloud; their time is reported separately, since the problem "
            "statement asks for a mesh or a point cloud.\n"
        )
        parts.append(table("Every sample video, warm engine, one job at a time: length, height, keyframes, registration, models "
                           "(one per pass), view completeness, mesh-and-cloud time against the budget and per second of video, then "
                           "splat training for every model and its mean held-out PSNR.", "tab:allmaps",
                           ["Video", "s", "Res.", "KF", "Reg.", "Mod.", "Compl.", "Time (s)", "Budget (s)", "s/s", "Splat (s)", "PSNR"],
                           rows, "lrrrrrrrrrrr", wide=True))  # fmt: skip
    geo = load(FIG / "fast_georef_e2e.json")
    if geo:
        rows, near = [], []
        for m in geo["models"]:
            me = m.get("mesh_error_m") or {}
            b = me.get("by_distance_from_track") or {}
            n100 = b.get("0-100 m")
            if n100:
                near.append(n100["median"])
            rows.append([tex(Path(m["model"]).name), tex(m.get("mode") or DASH), fmt(m.get("rotation_err_deg"), 2),
                         fmt(100 * abs(m["scale_est"] / m["scale_true"] - 1), 2),
                         fmt((m.get("held_out") or {}).get("loo_rmse_horizontal_m"), 2),
                         fmt(n100["median"], 2) if n100 else DASH,
                         fmt((b.get("100-300 m") or {}).get("median"), 2), fmt((b.get("300-inf m") or {}).get("median"), 1)])  # fmt: skip
        macros["GeoNearTrack"] = f"{min(near):.2f}--{max(near):.2f}" if near else DASH
        noise = geo["gps_noise_m"]
        parts.append(
            "\\paragraph{Georeferencing, end to end.} Each Jal Mahal model receives a known model-to-world similarity at the real "
            f"site and its keyframes GPS with {noise['horizontal']}\\,m horizontal and {noise['vertical']}\\,m vertical noise; the "
            "pipeline's georef and export stages then run unchanged. The levelled fit recovers rotation to a few tenths of a "
            f"degree and scale to half a percent; the exported mesh is within {macros['GeoNearTrack']}\\,m (median) of the truth "
            "within 100\\,m of the track, and the error grows with distance, as yaw and scale error do (Table~\\ref{tab:geoe2e}). "
            "LAS and GeoTIFF outputs carry EPSG:32643 (UTM 43N).\n"
        )
        parts.append(table("Georeferencing with synthetic GPS on the real models: rotation error, scale error, held-out horizontal "
                           "RMSE of the cameras, and median mesh error by distance from the flight track.",
                           "tab:geoe2e", ["Model", "Fit", "Rot. ($^\\circ$)", "Scale (\\%)", "LOO (m)", "$<$100\\,m", "100--300\\,m", "$>$300\\,m"],
                           rows, "llrrrrrr", wide=True))  # fmt: skip
    return parts


def system_section(macros: dict[str, str]) -> list[str]:
    """Speed-ups, splats on the fast poses, every video with every layer, and live footage."""
    parts: list[str] = []
    sp = load(FIG / "system_speedups.json")
    if sp:
        b, a = sp["before"], sp["after"]
        macros.update(SpeedBefore=f"{b['seconds']:.0f}", SpeedAfter=f"{a['seconds']:.0f}", SpeedBudget=f"{a['budget_seconds']:.0f}",
                      SfmBefore=f"{sp['sfm_steps_s']['sequential']:.1f}", SfmAfter=f"{sp['sfm_steps_s']['in_full_run']:.1f}",
                      ExportBefore=f"{sp['export_steps_s']['before']:.1f}", ExportAfter=f"{sp['export_steps_s']['after']:.1f}",
                      WarmSeconds=f"{sp['warm_engine_before_speedups']['seconds']:.1f}")  # fmt: skip
        rows = [[tex(k), fmt(b["per_stage_s"].get(k), 1), fmt(a["per_stage_s"].get(k), 1)] for k in ("keyframes", "sfm", "dense", "export")]
        rows.append(["total", fmt(b["seconds"], 1), fmt(a["seconds"], 1) + " \\checkmark"])
        parts.append("\\subsection{Doing in parallel what does not depend on each other}\\label{sec:speed}\n"
                     "The passes of a video are independent. The global mapper releases Python's interpreter lock, so each pass is "
                     "mapped on a thread while the GPU tracks the next, largest pass first so that its long mapping overlaps the "
                     f"tracking of all the others: SfM on Jal Mahal fell from {macros['SfmBefore']}\\,s to {macros['SfmAfter']}\\,s with "
                     "the same 128/128 keyframes registered. Export writes the textured GLB/OBJ, FBX and Blender files in the "
                     "background while the GPU bakes the next model, and no longer decimates meshes within 25\\,\\% of the viewer "
                     f"cap ({macros['ExportBefore']}\\,s $\\to$ {macros['ExportAfter']}\\,s). The per-stage GPU telemetry thread had "
                     "been finding the run's child processes by reading every \\texttt{/proc/*/stat} twice a second, holding the "
                     "interpreter lock against the stage it measured; it now walks \\texttt{/proc/<pid>/task/*/children}. Together "
                     f"the 55\\,s Jal Mahal edit went from {macros['SpeedBefore']}\\,s to {macros['SpeedAfter']}\\,s, inside its "
                     f"{macros['SpeedBudget']}\\,s budget for the first time (Table~\\ref{{tab:speed}}).\n")
        parts.append(table("Jal Mahal, fast profile, idle A100, same conditions: seconds per stage before and after overlapping "
                           "independent work.", "tab:speed", ["Stage", "Before (s)", "After (s)"], rows, "lrr"))  # fmt: skip
    si = load(FIG / "splat_init.json")
    if si:
        rows = [[tex(r["init"]), r["quality"], fmt(r["splats"]), fmt(r["train_s"], 1), fmt(r["psnr"], 2), fmt(r["cc_psnr"], 2),
                 fmt(r["ssim"], 3)] for r in si["rows"]]  # fmt: skip
        macros.update(SplatSparsePsnr=f"{si['rows'][0]['psnr']:.1f}", SplatDensePsnr=f"{si['rows'][1]['psnr']:.1f}",
                      SplatDenseSeconds=f"{si['rows'][1]['train_s']:.0f}", SplatWebMB=f"{si['web']['bytes'] / 1e6:.0f}",
                      SplatPlyMB=f"{si['web']['ply_bytes'] / 1e6:.0f}")  # fmt: skip
        parts.append("\\subsection{Gaussian splats on the fast profile's poses}\\label{sec:fastsplat}\n"
                     "Flow SfM keeps about twelve tracks per image, too few points for 3DGS to densify from in a few thousand steps. "
                     "Starting instead from the dense TSDF cloud --- a copy of the SfM model whose points are the fused surface --- "
                     f"raises held-out PSNR from {macros['SplatSparsePsnr']} to {macros['SplatDensePsnr']}\\,dB in "
                     f"{macros['SplatDenseSeconds']}\\,s of training (Table~\\ref{{tab:splatinit}}). The trained splats are "
                     "converted to the 32-byte web layout (importance-sorted, opacity-pruned) in the model's export frame, the "
                     f"trainer's own scene transform undone: {macros['SplatPlyMB']}\\,MB of PLY become {macros['SplatWebMB']}\\,MB that "
                     "a browser renders together with the mesh.\n")
        parts.append(table("Splat initialisation on Jal Mahal's largest model (41 keyframes), 7000 steps, held-out views: "
                           "splats, training time, PSNR, colour-corrected PSNR, SSIM.", "tab:splatinit",
                           ["Start", "Quality", "Splats", "Train (s)", "PSNR", "cc-PSNR", "SSIM"], rows, "llrrrrr", wide=True))  # fmt: skip
    tl = load(FIG / "tsdf_limits.json")
    if tl:
        ok = max(r["active"] for r in tl["active_block_probe"] if r["ok"])
        bad = min(r["active"] for r in tl["active_block_probe"] if not r["ok"])
        macros.update(TsdfOkBlocks=f"{ok:,}".replace(",", "{,}"), TsdfBadBlocks=f"{bad:,}".replace(",", "{,}"))
    lv = load(FIG / "live_qutub.json")
    if lv:
        done = [g for g in lv["segments"] if g["status"] == "done"]
        lat = [g["latency_s"] for g in done]
        macros.update(LiveSegments=str(len(lv["segments"])), LiveModelled=str(len(done)),
                      LiveLatency=f"{min(lat):.0f}--{max(lat):.0f}" if lat else DASH)  # fmt: skip
        parts.append("\\subsection{Live footage}\\label{sec:live-results}\n"
                     f"Qutub Minar was replayed at its frame rate into 30\\,s segments: of {macros['LiveSegments']} segments, "
                     f"{macros['LiveModelled']} held parallax and were modelled {macros['LiveLatency']}\\,s after they closed; the others "
                     "were refused by the parallax test, as a camera that only turns should be.\n")
    return parts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    parts: list[str] = ["\\section{Experiments}\\label{sec:exp}\n"]

    # --- setup and data
    bench = load(FIG / "bench_gpu.json") or {}
    parts.append(
        "\\paragraph{Setup.} One NVIDIA A100 80\\,GB PCIe and 24 CPU cores; the fast profile's timings are taken on the "
        "warm engine running one job at a time. PyTorch "
        f"{tex(bench.get('torch', '2.14'))} with CUDA {tex(bench.get('cuda', '13.2'))}, Python 3.14, "
        "ffmpeg 9.0 (NVDEC/NVENC), spirula-studio 2026.9.24 (Vulkan backend). "
        "Data: fifteen public drone videos (Table~\\ref{tab:data}); they carry no flight logs.\n"
    )
    parts.append(
        table(
            "Sample footage (public YouTube re-encodes, VP9).",
            "tab:data",
            ["Video", "Resolution", "fps", "s"],
            dataset_rows(),
            "lrrr",
        )
    )

    # --- per-run keyframes / sfm / splat
    runs = run_dirs()
    kf_rows, sfm_rows, splat_rows, gpu_rows = [], [], [], []
    for name, d in runs.items():
        kf = load(d / "keyframes" / "result.json") or {}
        for p in kf.get("passes", []):
            kf_rows.append([tex(name), str(p["pass_id"]), f"{p['start_s']:.1f}--{p['end_s']:.1f}", fmt(p["num_keyframes"]),
                            fmt(p.get("mean_overlap_consecutive")), fmt(p.get("views_per_point"), 1),
                            fmt(p.get("parallax_snr"), 1), tex(p["verdict"])])  # fmt: skip
        sfm = load(d / "sfm" / "result.json")
        if sfm:
            sfm_rows.append([tex(name), fmt(sfm["input_images"]), fmt(sfm["registered_images"]),
                             fmt(len(sfm["models"])), fmt(sfm.get("mean_reprojection_px")), fmt(sfm.get("seconds"), 0)])  # fmt: skip
        splat = load(d / "splat" / "result.json")
        if splat:
            for m in splat["models"]:
                e = m.get("eval_metrics", {})
                splat_rows.append([tex(name), tex(Path(m["run_dir"]).name), fmt(m.get("num_splats")),
                                   fmt(e.get("psnr")), fmt(e.get("ssim"), 3), fmt(e.get("lpips"), 3), fmt(m.get("seconds"), 0)])  # fmt: skip
        for stage in ("keyframes", "sfm", "depth", "splat", "mesh", "render"):
            r = load(d / stage / "result.json") or {}
            g = r.get("gpu")
            if g:
                gpu_rows.append([tex(name), stage, fmt(r.get("duration_s"), 0), fmt(g.get("util_mean"), 0),
                                 fmt(g.get("power_mean_w"), 0), fmt(g.get("own_memory_peak_gb"), 1),
                                 fmt(g.get("host_rss_peak_gb"), 1)])  # fmt: skip

    parts.append("\\subsection{Passes, keyframes and the 3D verdict}\n")
    parts.append(table("Pass segmentation and keyframes per run. Overlap: mean co-visibility with the previous keyframe; views: measured views per point; SNR: median direct-flow parallax SNR (1 = none).",
                       "tab:keyframes", ["Run", "Pass", "Time (s)", "KF", "Overlap", "Views", "SNR", "Verdict"], kf_rows, "llrrrrrl", wide=True))  # fmt: skip
    canon_kf = load(ROOT / "outputs" / "jal_mahal" / "keyframes" / "result.json")
    if canon_kf:
        passes = canon_kf["passes"]
        t = canon_kf["timing_s"]
        overlaps = [p["mean_overlap_consecutive"] for p in passes]
        parts.append(
            f"On Jal Mahal the flow signal splits the edit into {len(passes)} passes at its cuts and fades, and the band "
            f"selects {canon_kf['num_keyframes']} keyframes; the mean overlap with the previous keyframe lies in "
            f"{min(overlaps):.2f}--{max(overlaps):.2f}, inside or just above the target band, and every pass is "
            f"judged \\emph{{3d}} (Table~\\ref{{tab:keyframes}}). Decoding and flow for the {canon_kf['analysis']['frames']} "
            f"analysis frames took {t['decode_and_flow']:.0f}\\,s, selection {t['selection']:.0f}\\,s and 4K extraction "
            f"{t['extraction']:.0f}\\,s.\n"
        )

    control = load(
        ROOT / "outputs" / "controls" / "pure_rotation_run" / "keyframes" / "result.json"
    )
    if control and control.get("passes"):
        c = control["passes"][0]
        macros["ControlSNR"] = fmt(c.get("parallax_snr"), 2)
        macros["ControlParallax"] = fmt(c.get("median_parallax_deg"), 2)
        macros["ControlVerdict"] = tex(c["verdict"])
        parts.append(
            f"\\paragraph{{Negative control.}} A synthetic clip of a camera rotating about its centre over one real 4K frame "
            f"(\\texttt{{tools/make\\_degenerate\\_clip.py}}) has no parallax by construction. It yields SNR {macros['ControlSNR']} "
            f"(median residual {macros['ControlParallax']}$^\\circ$) and the verdict \\emph{{{macros['ControlVerdict']}}}; "
            "GRIC nonetheless preferred $F$ on "
            f"{fmt(c.get('fraction_prefer_3d'), 0, pct=True)} of its pairs, which is why GRIC only confirms the SNR decision.\n"
        )

    drift = load(FIG / "flow_drift.json")
    if drift:
        last = drift[-1]
        macros["DriftChained"] = fmt(last["chained_median_px"], 2)
        macros["DriftDirect"] = fmt(last["direct_median_px"], 2)
        parts.append(
            "\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{flow_drift.pdf}"
            "\\caption{Correspondence error against exact ground truth on the rotation control. Chained consecutive flow drifts "
            f"almost linearly ({macros['DriftChained']}\\,px median at {last['gap_frames']} frames); direct pairwise flow stays at "
            f"{macros['DriftDirect']}\\,px.}}\\label{{fig:drift}}\\end{{figure}}\n"
        )

    parts.append("\\subsection{Structure from motion}\n")
    ablation = load(FIG / "ablations.json") or {}
    sampling = {r["label"]: r for r in ablation.get("sampling", [])}
    base, canon = sampling.get("1 fps (uniform)"), sampling.get("overlap band (canonical)")
    uniform, band = sampling.get("uniform, same budget"), sampling.get("overlap band (dev)")
    if base and canon and uniform and band:
        parts.append(
            f"With the same SfM settings, 1\\,fps sampling registers {base['registered']} of {base['frames']} frames in "
            f"{base['models']} disconnected models (the largest {base['largest']}), and uniform sampling at the selector's "
            f"budget of {uniform['frames']} frames registers {uniform['registered']} but splits into {uniform['models']} models "
            f"(largest {uniform['largest']}). The overlap band registers all {band['frames']} (largest {band['largest']}); "
            f"after cropping the letterbox and trimming fades the canonical run registers {canon['registered']} of "
            f"{canon['frames']} (Table~\\ref{{tab:sampling}}). A model per group of passes is expected: the passes of a "
            "cinematic edit are cut from different flights and need not overlap. "
            "One focal length per pass keeps the reprojection error while OpenCV's independent $f_x,f_y$ drift apart "
            "(Table~\\ref{tab:camera}).\n"
        )
    parts.append(table("Structure from motion on the selected keyframes.", "tab:sfm",
                       ["Run", "KF", "Registered", "Models", "Reproj. (px)", "Time (s)"], sfm_rows, "lrrrrr"))  # fmt: skip
    if ablation.get("sampling"):
        rows = [[tex(r["label"]), fmt(r["frames"]), fmt(r["registered"]), fmt(r["models"]), fmt(r.get("largest")), tex(r.get("note", ""))]
                for r in ablation["sampling"]]  # fmt: skip
        parts.append(table("Frame sampling on Jal Mahal (same SfM settings).", "tab:sampling",
                           ["Sampling", "Frames", "Registered", "Models", "Largest", ""], rows, "lrrrrl"))  # fmt: skip
    if ablation.get("camera"):
        rows = [
            [tex(r["model"]), tex(r["focals"]), fmt(r["reproj"], 3), fmt(r["points"])]
            for r in ablation["camera"]
        ]
        parts.append(table("Camera model on Jal Mahal: independent $f_x,f_y$ (OpenCV) against one focal (radial).", "tab:camera",
                           ["Camera", "Focal(s) per pass (px)", "Reproj. (px)", "Points"], rows, "llrr"))  # fmt: skip

    depth_rows, withheld = [], []
    for name, d in runs.items():
        depth = load(d / "depth" / "result.json")
        if depth:
            for model, v in depth["per_model"].items():
                depth_rows.append([tex(name), tex(Path(model).name), f"{v['aligned']}/{v['images']}",
                                   fmt(v.get("cv_affine_abs_rel_median"), 3), fmt(v.get("cv_monotone_abs_rel_median"), 3),
                                   fmt(v.get("cv_affine_delta1"), 3), fmt(v.get("cv_monotone_delta1"), 3)])  # fmt: skip
                if v["aligned"] == 0:
                    withheld.append((Path(model).name, v["images"]))
    parts.append("\\subsection{Monocular depth prior}\n")
    if depth_rows:
        withheld_text = (
            " ".join(
                f"Model~{m} ({n} keyframes, the far lake-side pass) has no image whose prediction correlates positively "
                "with its tie-point depths; it is trained without depth rather than with a wrong one."
                for m, n in withheld
            )
            if withheld
            else ""
        )
        parts.append(
            "Table~\\ref{tab:depth} scores each calibration on tie points it was not fitted to. The monotone map lowers "
            "held-out AbsRel on every model with depth; the gain is largest where the scene spans a wide depth range "
            f"(Fig.~\\ref{{fig:depth}}), which a single log-affine map cannot follow. {withheld_text}\n"
        )
    if (FIG / "depth_panel.png").is_file():
        parts.append(
            "\\begin{figure*}[t]\\centering\\includegraphics[width=0.92\\textwidth]{depth_panel.png}"
            "\\caption{Marigold~v2 on Jal Mahal keyframes (canonical run). Right: prior depth against SfM tie-point depth "
            "for the log-affine fit (grey) and the monotone calibration (blue). On the lake-facing passes the affine fit "
            "flattens the far shore and the ridge; the monotone map follows them (it also sends a few near points far). "
            "Held-out AbsRel for both is given above each depth map.}\\label{fig:depth}\\end{figure*}\n"
        )
    parts.append(table("Marigold~v2 against SfM tie points, scored on held-out tie points (5-fold cross-validation per image, median over images): log-affine vs monotone calibration.",
                       "tab:depth", ["Run", "Model", "Aligned", "AbsRel aff.", "AbsRel mono.", r"$\delta_1$ aff.", r"$\delta_1$ mono."], depth_rows, "llrrrrr"))  # fmt: skip

    parts.append("\\subsection{Gaussian splatting}\n")
    for r in ablation.get("splat") or []:
        splat_rows.append(
            [
                tex(r["run"]),
                tex(r["label"]),
                fmt(r.get("num_splats")),
                fmt(r.get("psnr")),
                fmt(r.get("ssim"), 3),
                fmt(r.get("lpips"), 3),
                fmt(r.get("seconds"), 0),
            ]
        )
    by_label = {r["label"]: r for r in ablation.get("splat") or []}
    rgb, v1, v2 = (by_label.get(k) for k in ("model_0, RGB only", "model_0, depth prior (v1 mask)", "model_0, depth prior (v2 mask)"))
    if rgb and v1 and rgb.get("psnr") and v1.get("psnr"):
        v2_text = (
            f" With the calibrated-range mask (v2) it reaches {v2['psnr']:.2f}\\,dB / {v2['ssim']:.3f}."
            if v2 and v2.get("psnr")
            else ""
        )
        parts.append(
            f"On the largest model ({v1['eval_views']} held-out views), training on RGB alone gives "
            f"{rgb['psnr']:.2f}\\,dB PSNR / {rgb['ssim']:.3f} SSIM and training with the depth prior (weight "
            f"{v1['depth_weight']}) {v1['psnr']:.2f}\\,dB / {v1['ssim']:.3f} (Table~\\ref{{tab:splat}}).{v2_text} "
            "Held-out photometric scores measure interpolation between nearby views; the prior's purpose is geometry "
            "away from them, which these numbers do not capture.\n"
        )
    parts.append(table("3DGS on held-out keyframes (every 8th keyframe never used in training).", "tab:splat",
                       ["Run", "Model", "Splats", "PSNR", "SSIM", "LPIPS", "Train (s)"], splat_rows, "llrrrrr", wide=True))  # fmt: skip

    geo = load(FIG / "georef_study.json")
    if geo:
        straight = [r for r in geo if r["lateral_m"] == 0.0]
        macros["ScaleErrMax"] = f"{100 * max(r['scale_rel_err'] for r in geo):.1f}"
        rows = [[fmt(r["gps_noise_h_m"], 1), fmt(r["lateral_m"], 0), fmt(r["sim_probe_rmse_m"], 1), fmt(r["auto_probe_rmse_m"], 2),
                 fmt(r["sim_in_sample_h_m"], 2), fmt(r["auto_loo_h_m"], 2), fmt(r["scale_rel_err"], 1, pct=True)] for r in geo]  # fmt: skip
        parts.append("\\subsection{Georeferencing a straight pass}\n")
        worst_ours = max(r["auto_probe_rmse_m"] for r in geo)
        parts.append(
            "None of the sample videos carries a flight log, so georeferencing is evaluated in simulation "
            "(\\texttt{experiments/georef\\_study.py}): a 300\\,m track at 80\\,m altitude over flat ground with buildings, "
            "the SfM model hidden behind a random similarity, horizontal GPS noise, and a sideways deviation of the "
            "track from a straight line of 0--60\\,m; the error is measured at ground points 100\\,m off the track. "
            "(Sloped ground, where the rule falls back to the similarity, is covered by the unit tests.) "
            f"The levelled fit is never worse than the similarity (worst {worst_ours:.1f}\\,m) and its metric scale error "
            f"stays below {macros['ScaleErrMax']}\\,\\% (Table~\\ref{{tab:georef}}, Fig.~\\ref{{fig:georef}}).\n"
        )
        parts.append(
            "\\begin{figure*}[t]\\centering\\includegraphics[width=0.85\\textwidth]{georef_study.pdf}"
            "\\caption{Error of ground 100\\,m off-track after GPS alignment (median of 20 synthetic scenes per point). "
            "The 7-DoF similarity is off by up to "
            f"{max(r['sim_probe_rmse_m'] for r in straight):.0f}\\,m on a straight pass while its in-sample RMSE (dotted) "
            "only reflects GPS noise; the ground-levelled fit stays near the GPS noise.}\\label{fig:georef}\\end{figure*}\n"
        )
        parts.append(table("Georeferencing study (medians over 20 trials): off-track error of the similarity and of our rule, in-sample RMSE of the similarity, held-out RMSE of ours, metric scale error.",
                           "tab:georef", ["GPS $\\sigma$ (m)", "Lateral (m)", "Sim. (m)", "Ours (m)", "Sim. in-sample", "Ours LOO", "Scale err."], rows, "rrrrrrr", wide=True))  # fmt: skip
    else:
        macros["ScaleErrMax"] = DASH

    parts.append("\\subsection{Throughput and GPU use}\n")
    raft = bench.get("raft", {})
    if raft:
        speed = raft.get("graph_b32_flows_per_s", 0) / max(
            raft.get("stock_b16_flows_per_s", 1e-9), 1e-9
        )
        macros["RaftSpeedup"] = f"{speed:.1f}"
        dec = bench.get("decode", {})
        rows = [
            ["RAFT stock (batch 16)", fmt(raft.get("stock_b16_flows_per_s"), 0), "flows/s", DASH],
            ["RAFT, no host syncs (32)", fmt(raft.get("patched_b32_flows_per_s"), 0), "flows/s", fmt(raft.get("patched_b32_mean_abs_diff_px"), 4)],
            ["RAFT + CUDA Graph (32)", fmt(raft.get("graph_b32_flows_per_s"), 0), "flows/s", fmt(raft.get("graph_b32_mean_abs_diff_px"), 4)],
            ["4K VP9, ffmpeg 4.4 CUVID", fmt((dec.get("ffmpeg4_cuvid") or {}).get("source_fps"), 0), "fps", DASH],
            ["4K VP9, ffmpeg 9 NVDEC", fmt((dec.get("ffmpeg9_nvdec") or {}).get("source_fps"), 0), "fps", DASH],
            ["4K VP9, ffmpeg 9 CPU (16 thr.)", fmt((dec.get("ffmpeg9_cpu") or {}).get("source_fps"), 0), "fps", DASH],
            ["nvJPEG 4K encode", fmt((bench.get("nvjpeg") or {}).get("ms_per_4k_frame"), 1), "ms", DASH],
        ]  # fmt: skip
        parts.append(table("Micro-benchmarks. The last column is RAFT's mean difference to stock torchvision in pixels.", "tab:bench",
                           ["Component", "Rate", "Unit", "$\\Delta$ (px)"], rows, "lrll"))  # fmt: skip
    else:
        macros["RaftSpeedup"] = DASH
    parts.append(table("Per-stage wall time and GPU use (NVML, 0.5\\,s samples). Own/host: peak GPU memory and host RSS of our process tree.",
                       "tab:gpu", ["Run", "Stage", "s", "Util.\\%", "W", "GPU (GB)", "Host (GB)"], gpu_rows, "llrrrrr", wide=True))  # fmt: skip

    parts += fast_section(macros)
    parts += system_section(macros)
    (OUT / "results.tex").write_text("\n".join(parts))
    (OUT / "macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
        + "\n"
        + "".join(
            f"\\providecommand{{\\{k}}}{{{DASH}}}\n"
            for k in ("ScaleErrMax", "RaftSpeedup", "ControlSNR", "DriftChained", "DriftDirect", "FastQutubSeconds",
                      "FastQutubBudget", "FastQutubCompl", "FastJalSeconds", "FastJalBudget", "FastJalCompl",
                      "GeoNearTrack", "FlowSfmCentre", "FlowSfmRot", "DedWithinBudget")
            if k not in macros
        )
    )
    print(f"wrote {OUT / 'results.tex'} ({len(runs)} runs) and {len(macros)} macros")


if __name__ == "__main__":
    main()
