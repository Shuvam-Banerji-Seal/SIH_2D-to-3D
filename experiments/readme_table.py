"""Write the README's measured section from paper/figures/all_maps.json (and the throughput run, if any).

    uv run python experiments/collect_system.py && uv run python experiments/readme_table.py [--write]

Prints the Markdown; ``--write`` replaces the README between ``<!-- measured:start -->`` and
``<!-- measured:end -->``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"


def title(video: str) -> str:
    name = re.split(r"[｜|：:,]| - |\[|\(", video)[0].strip()
    return re.sub(r"\b(4K|4k|HD|Drone|Video|Cinematic)\b.*$", "", name).strip() or name


def section() -> str:
    rows = sorted(json.loads((FIG / "all_maps.json").read_text()), key=lambda r: r["seconds"])
    out = [
        "One NVIDIA A100 80 GB, the warm engine running one video at a time, budget = 1.5 × video length (the",
        "problem statement's 15 minutes for a 10-minute video). *Completeness* is the share of every registered",
        "keyframe's non-sky pixels whose ray hits the mesh; a model without a mesh counts as uncovered",
        "([`scene_view_completeness`](src/drone3d/metrics/quality.py)). *Mesh + cloud* is the fast profile",
        "(ingest to export); Gaussian splats for every model are trained afterwards and timed separately.",
        "",
        "| Video | Length | Keyframes | Registered | Models | Completeness | Mesh + cloud | Budget | Splats (all models) | Splat PSNR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ok = "✓" if r["fast_within_budget"] else "✗"
        splat = f"{r['splat_s']:.0f} s" if r.get("splat_s") else "—"
        psnr = f"{r['splat_psnr_mean']:.1f} dB" if r.get("splat_psnr_mean") else "—"
        comp = f"{r['completeness']:.2f}" if r.get("completeness") is not None else "—"
        out.append(f"| {title(r['video'])} | {r['seconds']:.0f} s | {r['keyframes']} | {r['registered']} | {r['models']} | {comp} | "
                   f"**{r['fast_s']:.0f} s** | {r['budget_s']:.0f} s {ok} | {splat} | {psnr} |")  # fmt: skip
    total_v = sum(r["seconds"] for r in rows)
    total_f = sum(r["fast_s"] for r in rows)
    within = sum(r["fast_within_budget"] for r in rows)
    comp = sorted(r["completeness"] for r in rows if r.get("completeness") is not None)
    med = (comp[len(comp) // 2] + comp[(len(comp) - 1) // 2]) / 2 if comp else None
    out += ["", f"{within} of {len(rows)} within the budget; {total_v / 60:.0f} min of footage at "
            f"{total_f / total_v:.2f} s per video second overall; median completeness {med:.2f}." if med is not None else ""]
    misses = [title(r["video"]) for r in rows if not r["fast_within_budget"]]
    if misses:
        out.append(f"Over the budget: {', '.join(misses)} — FPV flights, whose speed needs many keyframes per second, and clips "
                   "under half a minute, where fixed per-run costs dominate a budget proportional to length.")
    abl = FIG / "fast_ablation.json"
    if abl.is_file():
        fixed = [a for a in json.loads(abl.read_text()) if a["rate"].startswith("fixed") and a.get("completeness") is not None]
        if fixed:
            out += ["", "The rate trade-off is set in [`configs/fast.yaml`](configs/fast.yaml); analysing at a fixed 12 fps instead gives "
                    + ", ".join(f"{a['video']} {a['completeness']:.2f} / {a['seconds']:.0f} s" for a in fixed) + "."]
    thr = FIG / "engine_throughput.json"
    if thr.is_file():
        for t in json.loads(thr.read_text()):
            if t["slots"] > 1 and t.get("one_slot_sum_s"):
                out += ["", f"With {t['slots']} engine slots the same {t['videos']} videos took {t['wall_s']:.0f} s of wall time "
                        f"against {t['one_slot_sum_s']:.0f} s one at a time ({t['one_slot_sum_s'] / t['wall_s']:.2f}× the throughput): "
                        "one video's CPU phases (mapping, export) leave the GPU to another."]
    return "\n".join(out)


def main() -> None:
    text = section()
    print(text)
    if "--write" in sys.argv:
        readme = ROOT / "README.md"
        s = readme.read_text()
        a, b = "<!-- measured:start -->", "<!-- measured:end -->"
        if a not in s or b not in s:
            raise SystemExit("README has no measured markers")
        readme.write_text(s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):])


if __name__ == "__main__":
    main()
