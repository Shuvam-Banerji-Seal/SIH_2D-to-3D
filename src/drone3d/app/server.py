"""The drone3d console: configure and start reconstructions, follow them, explore and download results.

    uv run drone3d ui                 # http://127.0.0.1:8080

JSON under ``/api``: the configuration schema and profiles; videos and flight
logs (``datasets/``, ``uploads/``) and uploads; runs (start, list, status,
stop, their 3D scene, keyframe and depth filmstrips); live-stream sessions;
the warm engine (start/stop, models load/unload, capabilities); and the GPU,
sampled every second with a ten-minute history. Each run's files -- viewer,
report, downloads -- are served under ``/runs/<name>/``.

Reconstructions go to the warm engine when it is online and to a one-at-a-time
queue of ``drone3d run`` subprocesses otherwise, so the console works either
way; the web server itself never imports torch.
"""

from __future__ import annotations

import collections
import copy
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from drone3d.app.engine_client import EngineClient, EngineError
from drone3d.app.jobs import JobManager, run_status
from drone3d.app.schema import config_schema, profiles

__all__ = ["create_app"]

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}
LOG_EXT = {".srt", ".csv", ".gpx", ".json", ".tsv", ".txt"}
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SOURCE = re.compile(r"^(rtsp|rtmp|srt|udp|http|https|tcp)://\S+$|^/dev/video\d+$")


def _deep_set(d: dict, dotted: str, value: object) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class GpuSampler:
    """NVML once a second on a thread; the last ten minutes are kept for the console's charts."""

    def __init__(self, own_pids: Any, seconds: int = 600) -> None:
        self.own_pids = own_pids
        self.series: collections.deque[dict] = collections.deque(maxlen=seconds)
        self.latest: dict = {"available": False}
        threading.Thread(target=self._loop, name="gpu-sampler", daemon=True).start()

    def _loop(self) -> None:
        from drone3d.gpu.nvml import snapshot

        while True:
            try:
                own = self.own_pids()
                snap = snapshot(own)
                self.latest = snap
                if snap.get("available") and snap["gpus"]:
                    g = snap["gpus"][0]
                    own_mb = sum(p["used_mb"] for p in g["processes"] if p["own"])
                    self.series.append({"t": round(time.time(), 1), "util": g["util"], "mem_used_mb": g["mem_used_mb"],
                                        "mem_total_mb": g["mem_total_mb"], "own_mb": own_mb, "power_w": g["power_w"],
                                        "temp_c": g["temp_c"], "dec": g["decoder_util"], "enc": g["encoder_util"],
                                        "sm_mhz": g["sm_clock_mhz"]})  # fmt: skip
            except Exception:  # sampling must never stop the console
                pass
            time.sleep(1.0)


def _host() -> dict:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
    return {"cpus": os.cpu_count(), "load1": round(load[0], 2), "mem_total_gb": round(mem.get("MemTotal", 0) / 1e9, 1),
            "mem_available_gb": round(mem.get("MemAvailable", 0) / 1e9, 1)}  # fmt: skip


def create_app(repo: Path, outputs: Path | None = None, *, engine_port: int = 8770):  # type: ignore[no-untyped-def]
    from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    from drone3d.gpu.monitor import _descendants

    outputs = outputs or repo / "outputs"
    uploads = repo / "uploads"
    outputs.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    static = Path(__file__).parent / "static"
    viewer_static = Path(__file__).parent.parent / "viewer" / "static"
    jobs = JobManager(repo, outputs)
    engine = EngineClient(repo, outputs, port=engine_port)
    probe_cache: dict[str, dict] = {}

    def own_pids() -> set[int]:
        pids = {os.getpid()}
        info = _read(outputs / ".engine.json") or {}
        for pid in [info.get("pid"), jobs.current[1].pid if jobs.current else None]:
            if pid:
                pids |= {pid} | _descendants(pid)
        return pids

    gpu = GpuSampler(own_pids)
    app = FastAPI(title="drone3d console", docs_url="/api/docs")

    def fail(exc: EngineError) -> HTTPException:
        return HTTPException(exc.status, str(exc))

    # ------------------------------------------------------------- inputs
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

    def build_config(req: dict, *, need_video: bool = True) -> tuple[str, dict]:
        name = str(req.get("name") or "").strip() or time.strftime("run_%Y%m%d_%H%M%S")
        if not _NAME.match(name):
            raise HTTPException(400, "name: letters, digits, '.', '_' and '-' only")
        base = next(
            (p for p in profiles(repo / "configs") if p["name"] == req.get("profile")), None
        )
        if base is None:
            raise HTTPException(400, f"unknown profile {req.get('profile')!r}")
        config = copy.deepcopy(base["values"])
        for dotted, value in (req.get("overrides") or {}).items():
            _deep_set(config, dotted, value)
        if need_video:
            if not req.get("video"):
                raise HTTPException(400, "choose a video")
            _deep_set(config, "ingest.video", str(repo / req["video"]))
        if req.get("telemetry"):
            _deep_set(config, "ingest.telemetry", str(repo / req["telemetry"]))
        if req.get("stages"):
            config["stages"] = list(req["stages"])
        config["run_name"] = name
        return name, config

    # --------------------------------------------------------------- runs
    @app.get("/api/runs")
    def runs() -> list[dict]:
        items = []
        for d in outputs.iterdir():
            if (
                d.is_dir()
                and not d.name.startswith(".")
                and ((d / "logs" / "run.log").is_file() or (d / "ui_job.json").is_file())
            ):
                s = run_status(d)
                s.pop("log_tail", None)
                s["mtime"] = d.stat().st_mtime
                s["kind"] = (s.get("job") or {}).get("kind") or (
                    "live" if (d / "live.json").is_file() else "run"
                )
                items.append(s)
        return sorted(items, key=lambda s: -s["mtime"])

    @app.get("/api/runs/{name}")
    def run(name: str) -> dict:
        if not _NAME.match(name) or not (outputs / name).is_dir():
            raise HTTPException(404, "no such run")
        return {**run_status(outputs / name), "queue": jobs.queued()}

    @app.post("/api/runs")
    def start(req: dict = Body(...)) -> dict:  # noqa: B008
        name, config = build_config(req)
        if req.get("engine", True) and engine.online():
            try:
                job = engine.submit(name, config)
            except EngineError as exc:
                raise fail(exc) from exc
            return {"name": name, "via": "engine", "job": job}
        try:
            run_dir = jobs.submit(name, config)
        except Exception as exc:  # ConfigError / ValueError: show the pipeline's own message
            raise HTTPException(400, str(exc)) from exc
        return {"name": name, "via": "subprocess", "run_dir": str(run_dir), "queue": jobs.queued()}

    @app.post("/api/runs/{name}/stop")
    def stop(name: str) -> dict:
        job = _read(outputs / name / "ui_job.json") or {}
        if job.get("engine"):
            try:
                return {"stopped": bool(engine.cancel(name))}
            except EngineError as exc:
                raise fail(exc) from exc
        return {"stopped": jobs.stop(name)}

    def scene_of(run_dir: Path, url: str, label: str = "") -> list[dict]:
        scn = _read(run_dir / "export" / "scene.json") or {}
        rel = run_dir.relative_to(outputs)
        video = (_read(run_dir / "ingest" / "result.json") or {}).get("video") or {}
        out = []
        for m in scn.get("models", []):
            m = dict(m)
            for key in ("mesh", "points", "splat"):
                if m.get(key):
                    m[key] = f"{url}/export/{m[key]}"
            m["files"] = [{**f, "path": f"{url}/export/{f['path']}"} for f in m.get("files", [])]
            frame = (
                _read(run_dir / "export" / (m.get("dir") or "") / "frame.json")
                if m.get("dir")
                else None
            )
            m["frame"] = frame
            m["name"] = f"{label}{m.get('name', 'model')}"
            m["run"] = run_dir.name
            if m.get("frames"):
                for key in ("photo_dir", "depth_dir"):
                    m["frames"][key] = f"{url}/export/{m['frames'][key]}"
            # older exports have no frame previews: the console serves them from the run itself
            m.update(base=url, thumb=f"/api/thumb/{rel}", fps=video.get("fps"),
                     video=f"/api/video/{rel}" if video.get("path") else None)  # fmt: skip
            out.append(m)
        return out

    @app.get("/api/runs/{name}/scene")
    def scene(name: str) -> dict:
        d = outputs / name
        if not _NAME.match(name) or not d.is_dir():
            raise HTTPException(404, "no such run")
        live = _read(d / "live.json")
        if live:  # a live session: every finished segment's models, in order
            models = []
            for seg in live.get("segments", []):
                sd = Path(seg["run_dir"])
                if seg.get("status") == "done" and sd.is_dir():
                    rel = sd.relative_to(outputs)
                    models += scene_of(sd, f"/runs/{rel}", f"segment {seg['index']} · ")
            return {"title": name, "live": True, "models": models}
        title = (_read(d / "export" / "scene.json") or {}).get("title", name)
        return {"title": title, "live": False, "models": scene_of(d, f"/runs/{name}")}

    @app.get("/api/runs/{name}/frames")
    def frames(name: str, limit: int = Query(400, le=2000)) -> dict:
        """Keyframes with their fused depth preview (when the dense stage made one)."""
        d = outputs / name
        if not _NAME.match(name) or not d.is_dir():
            raise HTTPException(404, "no such run")
        runs_dirs = [d]
        live = _read(d / "live.json")
        if live:
            runs_dirs = [
                Path(s["run_dir"]) for s in live.get("segments", []) if Path(s["run_dir"]).is_dir()
            ]
        items = []
        for rd in runs_dirs:
            rel = rd.relative_to(outputs)
            depth = {p.stem: p for p in (rd / "dense").glob("model_*/depth/*.jpg")}
            for img in sorted((rd / "dataset" / "images").rglob("*.jpg")):
                items.append({"image": f"/api/thumb/{rel}/{img.relative_to(rd)}",
                              "full": f"/runs/{rel}/{img.relative_to(rd)}", "pass": img.parent.name,
                              "depth": f"/runs/{rel}/{depth[img.stem].relative_to(rd)}" if img.stem in depth else None})  # fmt: skip
        return {"frames": items[:limit], "total": len(items)}

    @app.get("/api/thumb/{path:path}")
    def thumb(path: str, w: int = Query(320, ge=64, le=960)) -> Response:
        src = (outputs / path).resolve()
        if (
            not src.is_relative_to(outputs.resolve())
            or not src.is_file()
            or src.suffix.lower() != ".jpg"
        ):
            raise HTTPException(404, "no such image")
        cache = src.parent / ".thumbs" / f"{src.stem}_{w}.jpg"
        if not cache.is_file() or cache.stat().st_mtime < src.stat().st_mtime:
            from PIL import Image

            cache.parent.mkdir(exist_ok=True)
            with Image.open(src) as im:
                im.thumbnail((w, w))
                im.convert("RGB").save(cache, "JPEG", quality=82)
        return FileResponse(
            cache, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"}
        )

    @app.get("/api/video/{rel:path}")
    def source_video(rel: str) -> FileResponse:
        """The video a run was made from (only that file: its path comes from the run's ingest result)."""
        run_dir = (outputs / rel).resolve()
        if not run_dir.is_relative_to(outputs.resolve()) or not run_dir.is_dir():
            raise HTTPException(404, "no such run")
        path = Path(((_read(run_dir / "ingest" / "result.json") or {}).get("video") or {}).get("path") or "")
        if not path.is_absolute():
            path = repo / path
        if not path.is_file():
            raise HTTPException(404, "the run's video is not on this machine")
        kind = {".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime"}.get(path.suffix.lower(), "video/mp4")
        return FileResponse(path, media_type=kind)

    # --------------------------------------------------------------- live
    @app.post("/api/live")
    def live_start(req: dict = Body(...)) -> dict:  # noqa: B008
        source = str(req.get("source") or "").strip()
        simulate = False
        if req.get("video"):  # rehearse with a recorded flight, replayed at its own frame rate
            source, simulate = str(repo / req["video"]), True
        elif not _SOURCE.match(source):
            raise HTTPException(
                400, "source: rtsp:// rtmp:// srt:// udp:// http(s):// tcp:// URL or /dev/videoN"
            )
        name, config = build_config(req, need_video=False)
        if not engine.online():
            try:
                engine.start()
            except EngineError as exc:
                raise fail(exc) from exc
        body = {"name": name, "source": source, "segment_s": float(req.get("segment_s") or 30), "simulate": simulate,
                "config": config, "telemetry": str(repo / req["telemetry"]) if req.get("telemetry") else None}  # fmt: skip
        try:
            return engine.live_start(body)
        except EngineError as exc:
            raise fail(exc) from exc

    @app.get("/api/live/{name}")
    def live_get(name: str) -> dict:
        state = _read(outputs / name / "live.json")
        if not _NAME.match(name) or state is None:
            raise HTTPException(404, "no such live session")
        return state

    @app.post("/api/live/{name}/stop")
    def live_stop(name: str) -> dict:
        try:
            return engine.live_stop(name)
        except EngineError as exc:
            raise fail(exc) from exc

    # ------------------------------------------------------------- engine
    @app.get("/api/engine")
    def engine_status() -> dict:
        return engine.status()

    @app.post("/api/engine/start")
    def engine_start(req: dict = Body(default={})) -> dict:  # noqa: B008
        try:
            return engine.start(warm=req.get("warm"))
        except EngineError as exc:
            raise fail(exc) from exc

    @app.post("/api/engine/stop")
    def engine_stop(req: dict = Body(default={})) -> dict:  # noqa: B008
        try:
            return engine.stop(force=bool(req.get("force")))
        except EngineError as exc:
            raise fail(exc) from exc

    @app.post("/api/engine/models/{key}/{action}")
    def engine_model(key: str, action: str) -> dict:
        if action not in ("load", "unload"):
            raise HTTPException(404, "load or unload")
        try:
            return engine.load(key) if action == "load" else engine.unload(key)
        except EngineError as exc:
            raise fail(exc) from exc

    @app.post("/api/engine/warm")
    def engine_warm(req: dict = Body(default={})) -> dict:  # noqa: B008
        from drone3d.engine.service import models_for

        keys = req.get("keys")
        if not keys and req.get("profile"):
            _, config = build_config({**req, "name": "warm"}, need_video=False)
            keys = models_for(config)
        try:
            return {"keys": keys, "result": engine.warm(keys=keys)}
        except EngineError as exc:
            raise fail(exc) from exc

    @app.post("/api/engine/unload")
    def engine_unload_all() -> dict:
        try:
            return engine.unload_all()
        except EngineError as exc:
            raise fail(exc) from exc

    @app.get("/api/engine/capabilities")
    def engine_caps() -> dict:
        try:
            return engine.capabilities()
        except EngineError as exc:
            raise fail(exc) from exc

    # ---------------------------------------------------------- telemetry
    @app.get("/api/gpu")
    def gpu_state(since: float = 0.0) -> dict:
        return {
            "now": time.time(),
            "latest": gpu.latest,
            "series": [s for s in gpu.series if s["t"] > since],
        }

    @app.get("/api/system")
    def system() -> dict:
        from drone3d.version import __version__

        du = shutil.disk_usage(outputs)
        return {"version": __version__, "disk_free_gb": round(du.free / 1e9, 1), "host": _host(),
                "queue": jobs.queued(), "engine_online": engine.online()}  # fmt: skip

    app.mount("/runs", StaticFiles(directory=str(outputs), html=True), name="runs")
    app.mount("/viewer", StaticFiles(directory=str(viewer_static)), name="viewer")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    return app
