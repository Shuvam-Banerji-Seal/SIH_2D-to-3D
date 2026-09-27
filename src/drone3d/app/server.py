"""The drone3d web app: configure and start reconstructions, follow them, explore and download results.

    uv run drone3d ui                 # http://127.0.0.1:8080

Endpoints (JSON under ``/api``): the configuration schema and profiles, videos
and flight logs available (``datasets/``, ``uploads/``), uploads, runs (start,
list, status, stop) and the GPU's state. Each run's files -- viewer, report,
downloads -- are served under ``/runs/<name>/``.
"""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
import time
from pathlib import Path

from drone3d.app.jobs import JobManager, run_status
from drone3d.app.schema import config_schema, profiles

__all__ = ["create_app"]

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}
LOG_EXT = {".srt", ".csv", ".gpx", ".json", ".tsv", ".txt"}
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _deep_set(d: dict, dotted: str, value: object) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def create_app(repo: Path, outputs: Path | None = None):  # type: ignore[no-untyped-def]
    from fastapi import Body, FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    outputs = outputs or repo / "outputs"
    uploads = repo / "uploads"
    outputs.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    static = Path(__file__).parent / "static"
    jobs = JobManager(repo, outputs)
    probe_cache: dict[str, dict] = {}
    app = FastAPI(title="drone3d", docs_url="/api/docs")

    def media(kind: set[str]) -> list[dict]:
        from drone3d.io.nvdec import probe_stream

        out = []
        for root, origin in ((repo / "datasets", "sample"), (uploads, "upload")):
            if not root.is_dir():
                continue
            for p in sorted(root.iterdir()):
                if p.suffix.lower() not in kind or not p.is_file():
                    continue
                entry = {"path": str(p.relative_to(repo)), "name": p.name, "origin": origin,
                         "size_mb": round(p.stat().st_size / 1e6, 1)}  # fmt: skip
                if kind is VIDEO_EXT:
                    key = f"{p}:{p.stat().st_mtime}"
                    if key not in probe_cache:
                        try:
                            info = probe_stream(p, count_frames=False)
                            probe_cache[key] = {"duration_s": round(info.duration_s, 1), "width": info.width,
                                                "height": info.height, "fps": round(info.fps, 2), "codec": info.codec}  # fmt: skip
                        except Exception:  # an unreadable file is listed without details
                            probe_cache[key] = {}
                    entry.update(probe_cache[key])
                out.append(entry)
        return out

    @app.get("/api/schema")
    def schema() -> dict:
        return config_schema()

    @app.get("/api/profiles")
    def get_profiles() -> list[dict]:
        return profiles(repo / "configs")

    @app.get("/api/videos")
    def videos() -> list[dict]:
        return media(VIDEO_EXT)

    @app.get("/api/telemetry")
    def telemetry() -> list[dict]:
        return media(LOG_EXT)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:  # noqa: B008
        name = Path(file.filename or "upload").name
        if Path(name).suffix.lower() not in VIDEO_EXT | LOG_EXT:
            raise HTTPException(400, f"unsupported file type: {name}")
        dest = uploads / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        return {
            "path": str(dest.relative_to(repo)),
            "name": name,
            "size_mb": round(dest.stat().st_size / 1e6, 1),
        }

    @app.get("/api/runs")
    def runs() -> list[dict]:
        items = []
        for d in outputs.iterdir():
            if d.is_dir() and ((d / "logs" / "run.log").is_file() or (d / "ui_job.json").is_file()):
                s = run_status(d)
                s.pop("log_tail", None)
                s["mtime"] = d.stat().st_mtime
                items.append(s)
        return sorted(items, key=lambda s: -s["mtime"])

    @app.get("/api/runs/{name}")
    def run(name: str) -> dict:
        if not _NAME.match(name) or not (outputs / name).is_dir():
            raise HTTPException(404, "no such run")
        return {**run_status(outputs / name), "queue": jobs.queued()}

    @app.post("/api/runs")
    def start(req: dict = Body(...)) -> dict:  # noqa: B008
        name = str(req.get("name") or "").strip() or time.strftime("run_%Y%m%d_%H%M%S")
        if not _NAME.match(name):
            raise HTTPException(400, "run name: letters, digits, '.', '_' and '-' only")
        base = next(
            (p for p in profiles(repo / "configs") if p["name"] == req.get("profile")), None
        )
        if base is None:
            raise HTTPException(400, f"unknown profile {req.get('profile')!r}")
        config = copy.deepcopy(base["values"])
        for dotted, value in (req.get("overrides") or {}).items():
            _deep_set(config, dotted, value)
        if not req.get("video"):
            raise HTTPException(400, "choose a video")
        _deep_set(config, "ingest.video", str(repo / req["video"]))
        if req.get("telemetry"):
            _deep_set(config, "ingest.telemetry", str(repo / req["telemetry"]))
        if req.get("stages"):
            config["stages"] = list(req["stages"])
        config["run_name"] = name
        try:
            run_dir = jobs.submit(name, config)
        except Exception as exc:  # ConfigError / ValueError: show the pipeline's own message
            raise HTTPException(400, str(exc)) from exc
        return {"name": name, "run_dir": str(run_dir), "queue": jobs.queued()}

    @app.post("/api/runs/{name}/stop")
    def stop(name: str) -> dict:
        return {"stopped": jobs.stop(name)}

    @app.get("/api/system")
    def system() -> dict:
        gpu = []
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)  # fmt: skip
            for line in q.stdout.strip().splitlines():
                n, used, total, util, temp, power = [x.strip() for x in line.split(",")]
                gpu.append({"name": n, "memory_used_mb": float(used), "memory_total_mb": float(total), "util_pct": float(util),
                            "temperature_c": float(temp), "power_w": float(power) if power not in ("", "[N/A]") else None})  # fmt: skip
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        du = shutil.disk_usage(outputs)
        from drone3d.version import __version__

        return {
            "gpu": gpu,
            "disk_free_gb": round(du.free / 1e9, 1),
            "version": __version__,
            "queue": jobs.queued(),
        }

    app.mount("/runs", StaticFiles(directory=str(outputs), html=True), name="runs")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    return app
