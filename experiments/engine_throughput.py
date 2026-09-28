"""Throughput of the warm engine with several slots: the sample videos' fast profile, all queued at once.

    uv run python experiments/engine_throughput.py SLOTS   (the engine at :8770 must run with SLOTS slots;
                                                             writes paper/figures/engine_throughput.json)

One video's pipeline leaves the GPU idle while it maps (GLOMAP, CPU) and exports (CPU); a
second slot fills those gaps with another video's GPU work. Wall time for the whole set
against the one-slot benchmark's sum of per-video times (paper/figures/all_maps.json).
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "http://127.0.0.1:8770"
FAST = ["ingest", "keyframes", "sfm", "dense", "georef", "export", "metrics", "report"]


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")  # fmt: skip
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def slug(name: str, n: int = 36) -> str:
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"\s*\[[^\]]+\]\.\w+$", "", name).lower()).strip("_")[:n].strip("_")


def main() -> None:
    slots = int(sys.argv[1])
    status = call("/status")
    if status["slots"] != slots:
        raise SystemExit(f"the engine runs {status['slots']} slot(s), not {slots}")
    base = yaml.safe_load((ROOT / "configs" / "fast.yaml").read_text())
    videos = sorted((p for p in (ROOT / "datasets").iterdir() if p.suffix == ".webm"), key=lambda p: p.stat().st_size)
    names = []
    for v in videos:
        name = f"thr{slots}_{slug(v.name)}"
        cfg = json.loads(json.dumps(base))
        cfg.update(ingest={"video": str(v)}, run_name=name, stages=FAST)
        call("/jobs", {"name": name, "config": cfg, "run_dir": str(ROOT / "outputs" / "experiments" / name)})
        names.append(name)
    while True:
        hist = {j["name"]: j for j in call("/status")["history"]}
        if all(n in hist and hist[n]["status"] in ("done", "failed", "stopped") for n in names):
            break
        time.sleep(15)
    jobs = [hist[n] for n in names]
    wall = max(j["finished"] for j in jobs) - min(j["started"] for j in jobs)
    video_s = sum(json.loads((ROOT / "outputs" / "experiments" / n / "ingest" / "result.json").read_text())["video"]["duration_s"] for n in names)
    one = {r["run"]: r for r in json.loads((ROOT / "paper" / "figures" / "all_maps.json").read_text())}
    keys = [f"map_{slug(v.name, 40)}" for v in videos]  # the one-slot benchmark's run names
    one_slot = sum(one[k]["fast_s"] for k in keys) if all(k in one for k in keys) else None
    row = {"slots": slots, "videos": len(names), "video_s": round(video_s, 1), "wall_s": round(wall, 1),
           "per_video_s": [round(j["seconds"], 1) for j in jobs], "failed": [j["name"] for j in jobs if j["status"] != "done"],
           "one_slot_sum_s": round(one_slot, 1) if one_slot else None}  # fmt: skip
    print(row)
    out = ROOT / "paper" / "figures" / "engine_throughput.json"
    rows = [r for r in (json.loads(out.read_text()) if out.is_file() else []) if r["slots"] != slots]
    out.write_text(json.dumps(sorted(rows + [row], key=lambda r: r["slots"]), indent=1))


if __name__ == "__main__":
    main()
