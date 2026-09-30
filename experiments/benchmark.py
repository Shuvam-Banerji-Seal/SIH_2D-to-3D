"""The all-videos benchmark: every sample video, fast profile + Gaussian splats, on the warm one-slot engine.

    uv run python experiments/benchmark.py [--no-wait] [--only KEYWORD ...] [--sequential [--tries N]]
    uv run python experiments/collect_system.py          (afterwards: paper/figures/all_maps.json)

Runs are written as ``outputs/map_<video slug>`` (replacing an earlier benchmark's). Timings are only
worth reporting on a GPU nobody else is computing on, so by default this waits until no other process
has held more than 3 GB of the card for two minutes (the host's idle 1.7 GB search server is ignored);
it never touches other processes. The engine at :8770 must run with one slot.

``--sequential`` submits one video at a time, each after the GPU has been quiet, samples the other
processes' load while it runs, and runs it again (up to ``--tries``) if the load was not quiet throughout:
another project's solvers came and went in bursts of an hour, and a batch submitted at once was timed
under them. Each video's load goes to ``outputs/.benchmark_load.json`` (``collect_system`` reads it).
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
# a crowd of small jobs passes the memory test: eight 1.4 GB diffraction solvers of another project held
# 90 % of the SMs together while the largest held 1.4 GB
FOREIGN_SM = 10
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


def foreign_sm(engine_pid: int) -> float:
    """Percent of the SMs that processes other than the engine and its children used, over three samples."""
    q = subprocess.run(["nvidia-smi", "pmon", "-c", "3", "-s", "u"], capture_output=True, text=True, check=True).stdout
    ours = _tree(engine_pid)
    total = 0
    for line in q.splitlines():
        f = line.split()
        if line.startswith("#") or len(f) < 4 or not f[1].isdigit() or not f[3].isdigit():
            continue
        if int(f[1]) not in ours:
            total += int(f[3])
    return total / 3


def wait_quiet() -> None:
    quiet_since = None
    while True:
        try:
            engine_pid = call("/status")["pid"]  # read each time: the engine may have been restarted meanwhile
        except OSError:
            time.sleep(15)
            continue
        busy, sm = foreign_gb(engine_pid), foreign_sm(engine_pid)
        now = time.time()
        if busy > FOREIGN_GB or sm > FOREIGN_SM:
            if quiet_since is not None or not hasattr(wait_quiet, "said"):
                print(f"{time.strftime('%H:%M:%S')} other processes hold {busy:.1f} GB (largest) and {sm:.0f} % of the SMs; waiting", flush=True)
                wait_quiet.said = True  # type: ignore[attr-defined]
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= QUIET_S:
            print(f"{time.strftime('%H:%M:%S')} GPU quiet for {QUIET_S} s", flush=True)
            return
        time.sleep(15)


LOAD = ROOT / "outputs" / ".benchmark_load.json"


def run_alone(name: str, cfg: dict, tries: int) -> dict:
    """One video on a quiet GPU -> its record: status, seconds and the other processes' load while it ran."""
    rec: dict = {}
    for attempt in range(1, tries + 1):
        wait_quiet()
        since = time.time()
        call("/jobs", {"name": name, "config": cfg, "run_dir": str(ROOT / "outputs" / name)})
        gb, sm = [], []
        while True:
            st = call("/status")
            job = next((j for j in st["history"] if j["name"] == name and j.get("submitted", 0) >= since), None)
            if job is not None and job["status"] in ("done", "failed", "stopped"):
                break
            gb.append(foreign_gb(st["pid"]))
            sm.append(foreign_sm(st["pid"]))
            time.sleep(20)
        rec = {"status": job["status"], "seconds": job.get("seconds"), "attempt": attempt, "samples": len(sm),
               "foreign_gb_peak": round(max(gb, default=0.0), 2), "foreign_sm_mean": round(sum(sm) / max(len(sm), 1), 1),
               "foreign_sm_peak": round(max(sm, default=0.0), 1)}  # fmt: skip
        rec["clean"] = rec["foreign_gb_peak"] <= FOREIGN_GB and rec["foreign_sm_mean"] <= FOREIGN_SM / 2
        print(f"{time.strftime('%H:%M:%S')} {name} {rec['status']} {rec['seconds']} s  others: {rec['foreign_sm_mean']} % SM "
              f"(peak {rec['foreign_sm_peak']}), {rec['foreign_gb_peak']} GB{'' if rec['clean'] else '  -- shared, again'}", flush=True)
        if rec["clean"] or rec["status"] != "done":
            break
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true", help="submit at once (timings may then be shared)")
    ap.add_argument("--only", nargs="*", default=None, help="videos whose file name contains one of these")
    ap.add_argument("--sequential", action="store_true", help="one video at a time, each re-run if the GPU was shared")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()
    status = call("/status")
    if status["slots"] != 1:
        raise SystemExit(f"the engine runs {status['slots']} slots; the benchmark is timed on one")
    videos = sorted((p for p in (ROOT / "datasets").iterdir() if p.suffix == ".webm"), key=lambda p: p.name.lower())
    if args.only:
        videos = [v for v in videos if any(k.lower() in v.name.lower() for k in args.only)]
    if not args.no_wait and not args.sequential:
        wait_quiet()
    base = yaml.safe_load((ROOT / "configs" / "fast.yaml").read_text())
    names = []
    since = time.time()  # the engine keeps its history: a run of the same name before this one is not ours
    for v in videos:
        name = f"map_{slug(v.name)}"
        cfg = json.loads(json.dumps(base))
        cfg.update(ingest={**(cfg.get("ingest") or {}), "video": str(v)}, run_name=name, stages=STAGES)
        cfg["splat"] = {**(cfg.get("splat") or {}), **SPLAT}
        if args.sequential:
            rec = run_alone(name, cfg, args.tries)
            load = json.loads(LOAD.read_text()) if LOAD.is_file() else {}
            load[name] = {**rec, "measured": time.strftime("%Y-%m-%d %H:%M")}
            LOAD.write_text(json.dumps(load, indent=1))
        else:
            call("/jobs", {"name": name, "config": cfg, "run_dir": str(ROOT / "outputs" / name)})
            print("queued", name, flush=True)
        names.append(name)
    while True:
        hist = {j["name"]: j for j in reversed(call("/status")["history"]) if j.get("submitted", 0) >= since}  # newest wins
        if all(n in hist and hist[n]["status"] in ("done", "failed", "stopped") for n in names):
            break
        time.sleep(30)
    for n in names:
        print(n, hist[n]["status"], hist[n].get("seconds"), flush=True)
    sys.exit(0 if all(hist[n]["status"] == "done" for n in names) else 1)


if __name__ == "__main__":
    main()
