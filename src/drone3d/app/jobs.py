"""Runs started from the UI: a one-at-a-time queue of ``drone3d run`` subprocesses, and run status.

GPU stages must not share the card with each other, so jobs run in submission
order. Each job's state lives in ``<run>/ui_job.json`` and its progress is read
from the pipeline's own ``logs/run.log`` -- the same files the CLI writes, so a
run started from the terminal shows up in the UI too.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

__all__ = ["JobManager", "run_status"]

_STAGE_START = re.compile(
    r"^(\d\d:\d\d:\d\d) \| \w+\s*\| drone3d\.pipeline \| === stage: (\w+) ==="
)
_STAGE_DONE = re.compile(
    r"^(\d\d:\d\d:\d\d) \| \w+\s*\| drone3d\.pipeline \| stage (\w+): (\w+) \(([\d.]+)s\) ?(.*)$"
)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_status(run_dir: Path) -> dict:
    """Everything the UI shows about a run, from its files."""
    job = _read(run_dir / "ui_job.json") or {}
    cfg = {}
    if (run_dir / "ui_config.yaml").is_file():
        cfg = yaml.safe_load((run_dir / "ui_config.yaml").read_text()) or {}
    stages_planned = cfg.get("stages") or []
    stages: dict[str, dict] = {s: {"status": "pending"} for s in stages_planned}
    log = run_dir / "logs" / "run.log"
    current, tail = None, []
    if log.is_file():
        lines = log.read_text(errors="replace").splitlines()
        tail = lines[-60:]
        for line in lines:
            if m := _STAGE_START.match(line):
                stages.setdefault(m.group(2), {})["status"] = "running"
                stages[m.group(2)]["started"] = m.group(1)
                current = m.group(2)
            elif m := _STAGE_DONE.match(line):
                stages.setdefault(m.group(2), {}).update(
                    status=m.group(3), seconds=float(m.group(4)), message=m.group(5)
                )
                if current == m.group(2):
                    current = None
    status = job.get("status") or ("done" if (run_dir / "manifest.json").is_file() else "unknown")
    if status == "running" and job.get("pid") and not _alive(job["pid"]):
        status = "done" if job.get("exit_code") == 0 else "stopped"
    metrics = _read(run_dir / "metrics" / "metrics.json") or {}
    ingest = _read(run_dir / "ingest" / "result.json") or {}
    sfm = _read(run_dir / "sfm" / "result.json") or {}
    dense = _read(run_dir / "dense" / "result.json") or {}
    export = _read(run_dir / "export" / "result.json") or {}
    geo = _read(run_dir / "georef" / "result.json") or {}
    comp = [
        m["view_completeness"]
        for m in dense.get("models", [])
        if m.get("view_completeness") is not None
    ]
    video = ingest.get("video") or {}
    summary = {
        "video": Path(video.get("path", "")).name or (cfg.get("ingest") or {}).get("video"),
        "video_seconds": video.get("duration_s"),
        "resolution": [video.get("width"), video.get("height")] if video else None,
        "keyframes": sfm.get("input_images"),
        "registered": sfm.get("registered_images"),
        "models": len(sfm.get("models", [])) or None,
        "triangles": sum(m.get("mesh_triangles") or 0 for m in dense.get("models", []) if m.get("status") == "ok") or None,
        "completeness": round(sum(comp) / len(comp), 3) if comp else None,
        "processing": metrics.get("processing"),
        "georeferenced": bool(geo.get("models")),
        "georef": [{"model": Path(m["model"]).name, "mode": m.get("mode"),
                    "loo_rmse_horizontal_m": (m.get("held_out") or {}).get("loo_rmse_horizontal_m")} for m in geo.get("models", [])],
        "viewer": (run_dir / "export" / "index.html").is_file(),
        "report": (run_dir / "report.html").is_file(),
        "files": [{"model": Path(m["model"]).name, "files": m["files"], "epsg": m.get("epsg"),
                   "triangles": m.get("triangles"), "points": m.get("points")} for m in export.get("models", [])],
    }  # fmt: skip
    return {"name": run_dir.name, "status": status, "job": job, "config": cfg, "stages": stages, "current": current,
            "summary": summary, "log_tail": tail}  # fmt: skip


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class JobManager:
    """Queue of pipeline runs; one runs at a time."""

    def __init__(self, repo: Path, outputs: Path) -> None:
        self.repo, self.outputs = repo, outputs
        self.queue: list[str] = []
        self.current: tuple[str, subprocess.Popen] | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, name: str, config: dict) -> Path:
        from drone3d.config import (  # noqa: PLC0415  (validate with the pipeline's own rules)
            PipelineConfig,
            _build,
        )

        cfg = _build(PipelineConfig, config)
        cfg.validate()
        run_dir = self.outputs / name
        if run_dir.exists() and (run_dir / "ui_job.json").is_file():
            state = (_read(run_dir / "ui_job.json") or {}).get("status")
            if state in ("queued", "running"):
                raise ValueError(f"run {name!r} is already {state}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ui_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        self._state(run_dir, status="queued", submitted=time.time())
        with self.lock:
            self.queue.append(name)
        return run_dir

    def stop(self, name: str) -> bool:
        with self.lock:
            if name in self.queue:
                self.queue.remove(name)
                self._state(self.outputs / name, status="stopped")
                return True
            if self.current and self.current[0] == name:
                proc = self.current[1]
                try:
                    os.killpg(
                        proc.pid, signal.SIGTERM
                    )  # the job's own process group: pipeline + its children
                except OSError:
                    proc.terminate()
                self._state(self.outputs / name, status="stopping")
                return True
        return False

    def queued(self) -> list[str]:
        with self.lock:
            return list(self.queue)

    def _state(self, run_dir: Path, **kw) -> None:  # type: ignore[no-untyped-def]
        state = _read(run_dir / "ui_job.json") or {}
        state.update(kw)
        (run_dir / "ui_job.json").write_text(json.dumps(state, indent=1))

    def _loop(self) -> None:
        while True:
            with self.lock:
                name = self.queue.pop(0) if self.queue and self.current is None else None
            if name is None:
                time.sleep(1.0)
                continue
            run_dir = self.outputs / name
            exe = Path(sys.executable).with_name("drone3d")
            cmd = [
                str(exe),
                "run",
                "--config",
                str(run_dir / "ui_config.yaml"),
                "--run-dir",
                str(run_dir),
            ]
            env = {**os.environ}
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if Path("/usr/local/cuda-13.3").is_dir():
                env.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")
            with (run_dir / "ui_stdout.log").open("w") as out:
                proc = subprocess.Popen(cmd, cwd=self.repo, stdout=out, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True)  # fmt: skip
            with self.lock:
                self.current = (name, proc)
            self._state(run_dir, status="running", pid=proc.pid, started=time.time())
            code = proc.wait()
            prev = (_read(run_dir / "ui_job.json") or {}).get("status")
            self._state(run_dir, status="stopped" if prev == "stopping" else ("done" if code == 0 else "failed"),
                        exit_code=code, finished=time.time())  # fmt: skip
            with self.lock:
                self.current = None
