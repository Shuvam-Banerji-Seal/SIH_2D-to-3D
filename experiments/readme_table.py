"""Print the README's benchmark table (Markdown) from paper/figures/all_maps.json.

    uv run python experiments/collect_system.py && uv run python experiments/readme_table.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"


def title(video: str) -> str:
    name = re.split(r"[｜|：:,]| - |\[|\(", video)[0].strip()
    return re.sub(r"\b(4K|4k|HD|Drone|Video|Cinematic)\b.*$", "", name).strip() or name


def main() -> None:
    rows = sorted(json.loads((FIG / "all_maps.json").read_text()), key=lambda r: r["seconds"])
    print("| Video | Length | Keyframes | Models | Completeness | Mesh + cloud | Budget | Splats (all models) | Splat PSNR |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ok = "✓" if r["fast_within_budget"] else "✗"
        splat = f"{r['splat_s']:.0f} s" if r.get("splat_s") else "—"
        psnr = f"{r['splat_psnr_mean']:.1f} dB" if r.get("splat_psnr_mean") else "—"
        print(f"| {title(r['video'])} | {r['seconds']:.0f} s | {r['keyframes']} | {r['models']} | {r['completeness']:.2f} | "
              f"**{r['fast_s']:.0f} s** | {r['budget_s']:.0f} s {ok} | {splat} | {psnr} |")  # fmt: skip
    total_v = sum(r["seconds"] for r in rows)
    total_f = sum(r["fast_s"] for r in rows)
    within = sum(r["fast_within_budget"] for r in rows)
    print(f"\n{within} of {len(rows)} within the budget; {total_v / 60:.0f} min of footage at {total_f / total_v:.2f} s per video second overall.")


if __name__ == "__main__":
    main()
