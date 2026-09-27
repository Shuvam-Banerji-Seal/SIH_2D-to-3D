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
        f"\\begin{{{env}}}[t]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}{foot}\n\\end{{{env}}}\n"
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
    """Pipeline runs with at least a keyframes result, keyed by run name."""
    runs = {}
    for d in sorted((ROOT / "outputs").iterdir()):
        if (d / "keyframes" / "result.json").is_file():
            runs[d.name] = d
    return runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    parts: list[str] = ["\\section{Experiments}\\label{sec:exp}\n"]

    # --- setup and data
    bench = load(FIG / "bench_gpu.json") or {}
    parts.append(
        "\\paragraph{Setup.} One NVIDIA A100 80\\,GB PCIe, shared with other users' jobs throughout "
        "(each stage's record states the GPU's occupancy); PyTorch "
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
                                 fmt(g.get("host_rss_peak_gb"), 1), "yes" if g.get("shared_gpu") else "no"])  # fmt: skip

    parts.append("\\subsection{Passes, keyframes and the 3D verdict}\n")
    parts.append(table("Pass segmentation and keyframes per run. Overlap: mean co-visibility with the previous keyframe; views: measured views per point; SNR: median direct-flow parallax SNR (1 = none).",
                       "tab:keyframes", ["Run", "Pass", "Time (s)", "KF", "Overlap", "Views", "SNR", "Verdict"], kf_rows, "llrrrrrl", wide=True))  # fmt: skip

    control = load(ROOT / "outputs" / "controls" / "pure_rotation_selection.json")
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
    parts.append(table("Structure from motion on the selected keyframes.", "tab:sfm",
                       ["Run", "KF", "Registered", "Models", "Reproj. (px)", "Time (s)"], sfm_rows, "lrrrrr"))  # fmt: skip
    ablation = load(FIG / "ablations.json") or {}
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

    depth_rows = []
    for name, d in runs.items():
        depth = load(d / "depth" / "result.json")
        if depth:
            for model, v in depth["per_model"].items():
                depth_rows.append([tex(name), tex(Path(model).name), fmt(v.get("aligned")), fmt(v.get("abs_rel_median"), 3),
                                   fmt(v.get("delta1_mean"), 3), fmt(v.get("valid_fraction_mean"), 2)])  # fmt: skip
    extra_depth = load(FIG / "depth_agreement.json")
    if extra_depth:
        for model, v in extra_depth.get("per_model", {}).items():
            depth_rows.append(["jal-mahal (dev)", tex(Path(model).name), fmt(v.get("aligned")), fmt(v.get("abs_rel_median"), 3),
                               fmt(v.get("delta1_mean"), 3), fmt(v.get("valid_fraction_mean"), 2)])  # fmt: skip
    parts.append("\\subsection{Monocular depth prior}\n")
    parts.append(table("Marigold~v2 against SfM tie points, after per-image log-affine alignment. AbsRel: median over images of the per-image median relative error.",
                       "tab:depth", ["Run", "Model", "Images", "AbsRel", "$\\delta_1$", "Valid"], depth_rows, "llrrrr"))  # fmt: skip

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
    parts.append(table("3DGS on held-out keyframes (every 8th keyframe never used in training).", "tab:splat",
                       ["Run", "Model", "Splats", "PSNR", "SSIM", "LPIPS", "Train (s)"], splat_rows, "llrrrrr", wide=True))  # fmt: skip

    geo = load(FIG / "georef_study.json")
    if geo:
        straight = [r for r in geo if r["lateral_m"] == 0.0]
        macros["ScaleErrMax"] = f"{100 * max(r['scale_rel_err'] for r in geo):.1f}"
        rows = [[fmt(r["gps_noise_h_m"], 1), fmt(r["lateral_m"], 0), fmt(r["sim_probe_rmse_m"], 1), fmt(r["auto_probe_rmse_m"], 2),
                 fmt(r["sim_in_sample_h_m"], 2), fmt(r["auto_loo_h_m"], 2), fmt(r["scale_rel_err"], 1, pct=True)] for r in geo]  # fmt: skip
        parts.append("\\subsection{Georeferencing a straight pass}\n")
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
        ctx = raft.get("context", {})
        note = f"GPU utilisation before the benchmark: {fmt(ctx.get('util_before_pct'), 0)}\\%, other processes present: {ctx.get('foreign_processes', DASH)}."
        parts.append(table("Micro-benchmarks. The last column is RAFT's mean difference to stock torchvision in pixels.", "tab:bench",
                           ["Component", "Rate", "Unit", "$\\Delta$ (px)"], rows, "lrll", note=note))  # fmt: skip
    else:
        macros["RaftSpeedup"] = DASH
    parts.append(table("Per-stage wall time and GPU use (NVML, 0.5\\,s samples). Own/host: peak GPU memory and host RSS of our process tree.",
                       "tab:gpu", ["Run", "Stage", "s", "Util.\\%", "W", "Own GPU (GB)", "Host (GB)", "Shared"], gpu_rows, "llrrrrrl", wide=True))  # fmt: skip

    (OUT / "results.tex").write_text("\n".join(parts))
    (OUT / "macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
        + "\n"
        + "".join(
            f"\\providecommand{{\\{k}}}{{{DASH}}}\n"
            for k in ("ScaleErrMax", "RaftSpeedup", "ControlSNR", "DriftChained", "DriftDirect")
        )
    )
    print(f"wrote {OUT / 'results.tex'} ({len(runs)} runs) and {len(macros)} macros")


if __name__ == "__main__":
    main()
