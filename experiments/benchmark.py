"""The all-videos benchmark: every sample video, fast profile + Gaussian splats, on the warm one-slot engine.

    uv run python experiments/benchmark.py [--no-wait] [--only KEYWORD ...]
    uv run python experiments/collect_system.py          (afterwards: paper/figures/all_maps.json)

Runs are written as ``outputs/map_<video slug>`` (replacing an earlier benchmark's). Timings are only
worth reporting on a GPU nobody else is computing on, so by default this waits until no other process
has held more than 3 GB of the card for two minutes (the host's idle 1.7 GB search server is ignored);
it never touches other processes. The engine at :8770 must run with one slot.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "http://127.0.0.1:8770"
STAGES = ["ingest", "keyframes", "sfm", "dense", "georef", "splat", "export", "metrics", "report"]
SPLAT = {"models": "all", "quality": "medium", "iterations": 7000, "depth_weight": 0, "parallel": 1}
FOREIGN_GB = 3.0
QUIET_S = 120


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")  # fmt: skip
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def slug(name: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"\s*\[[^\]]+\]\.\w+$", "", name).lower()).strip("_")[:n].strip("_")


def _tree(pid: int) -> set[int]:
    out, todo = {pid}, [pid]
    while todo:
        p = todo.pop()
        try:
            kids = Path(f"/proc/{p}/task/{p}/children").read_text().split()
        except OSError:
            continue
        for k in map(int, kids):
            if k not in out:
                out.add(k)
                todo.append(k)
    return out


def foreign_gb(engine_pid: int) -> float:
    """GPU memory held by the largest compute process that is not the engine or one of its children."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout  # fmt: skip
    ours = _tree(engine_pid)
    used = [int(m) / 1024 for p, m in (line.split(", ") for line in q.strip().splitlines() if line) if int(p) not in ours]
    return max(used, default=0.0)


def wait_quiet() -> None:
    quiet_since = None
    while True:
        try:
            engine_pid = call("/status")["pid"]  # read each time: the engine may have been restarted meanwhile
        except OSError:
            time.sleep(15)
            continue
        busy = foreign_gb(engine_pid)
        now = time.time()
        if busy > FOREIGN_GB:
            if quiet_since is not None or not hasattr(wait_quiet, "said"):
                print(f"{time.strftime('%H:%M:%S')} another process holds {busy:.1f} GB of the GPU; waiting", flush=True)
                wait_quiet.said = True  # type: ignore[attr-defined]
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= QUIET_S:
            print(f"{time.strftime('%H:%M:%S')} GPU quiet for {QUIET_S} s", flush=True)
            return
        time.sleep(15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true", help="submit at once (timings may then be shared)")
    ap.add_argument("--only", nargs="*", default=None, help="videos whose file name contains one of these")
    args = ap.parse_args()
    status = call("/status")
    if status["slots"] != 1:
        raise SystemExit(f"the engine runs {status['slots']} slots; the benchmark is timed on one")
    videos = sorted((p for p in (ROOT / "datasets").iterdir() if p.suffix == ".webm"), key=lambda p: p.name.lower())
    if args.only:
        videos = [v for v in videos if any(k.lower() in v.name.lower() for k in args.only)]
    if not args.no_wait:
        wait_quiet()
    base = yaml.safe_load((ROOT / "configs" / "fast.yaml").read_text())
    names = []
    since = time.time()  # the engine keeps its history: a run of the same name before this one is not ours
    for v in videos:
        name = f"map_{slug(v.name)}"
        cfg = json.loads(json.dumps(base))
        cfg.update(ingest={**(cfg.get("ingest") or {}), "video": str(v)}, run_name=name, stages=STAGES)
        cfg["splat"] = {**(cfg.get("splat") or {}), **SPLAT}
        call("/jobs", {"name": name, "config": cfg, "run_dir": str(ROOT / "outputs" / name)})
        names.append(name)
        print("queued", name, flush=True)
    while True:
        hist = {j["name"]: j for j in call("/status")["history"] if j.get("submitted", 0) >= since}
        if all(n in hist and hist[n]["status"] in ("done", "failed", "stopped") for n in names):
            break
        time.sleep(30)
    for n in names:
        print(n, hist[n]["status"], hist[n].get("seconds"), flush=True)
    sys.exit(0 if all(hist[n]["status"] == "done" for n in names) else 1)


if __name__ == "__main__":
    main()
