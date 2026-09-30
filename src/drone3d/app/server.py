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
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import (  # module level: the handlers' annotations resolve here
    Body,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from drone3d.app.engine_client import EngineClient, EngineError
from drone3d.app.jobs import JobManager, run_status
from drone3d.app.schema import config_schema, profiles

__all__ = ["create_app"]

log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}
LOG_EXT = {".srt", ".csv", ".gpx", ".json", ".tsv", ".txt"}
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def model_catalog(run: Path) -> dict:
    """The run's models, largest first: shots, keyframes, completeness, triangles, splat quality, and a role.

    A model is a *main* piece when it holds at least 15 % of the registered keyframes (the largest always
    is), a *fragment* otherwise -- a short shot nothing else overlaps; fragments are kept, not shown first.
    """
    sfm, dense = _read(run / "sfm" / "result.json") or {}, _read(run / "dense" / "result.json") or {}
    splat = _read(run / "splat" / "result.json") or {}
    d_by = {Path(m["model"]).name: m for m in dense.get("models", [])}
    s_by = {Path(m["model"]).name: m for m in splat.get("models", []) if m.get("model")}
    total = sum(m.get("images") or 0 for m in sfm.get("models", [])) or 1
    out = []
    for m in sfm.get("models", []):
        k = Path(m["path"]).name
        d, sp = d_by.get(k, {}), s_by.get(k, {})
        photos = sorted((run / "export" / f"model_{k}" / "frames" / "photo").glob("*.jpg"))
        ev = sp.get("eval_metrics") or {}
        psnr = ev.get("avg_psnr", ev.get("psnr"))
        if isinstance(psnr, list):
            psnr = sum(psnr) / len(psnr) if psnr else None
        out.append({"model": k, "images": m.get("images"), "share": round((m.get("images") or 0) / total, 3),
                    "shots": sorted(m.get("passes") or {}), "merged_from": m.get("merged_from"),
                    "completeness": d.get("view_completeness"), "triangles": d.get("mesh_triangles"),
                    "status": d.get("status"), "splat_psnr": round(psnr, 2) if isinstance(psnr, (int, float)) else None,
                    "splats": sp.get("num_splats"),
                    "thumb": f"/runs/{run.name}/export/model_{k}/frames/photo/{photos[len(photos) // 2].name}" if photos else None,
                    "glb": f"model_{k}/mesh_textured.glb" if (run / "export" / f"model_{k}" / "mesh_textured.glb").is_file() else None})  # fmt: skip
    out.sort(key=lambda e: -(e["images"] or 0))
    for i, e in enumerate(out):
        e["role"] = "main" if i == 0 or e["share"] >= 0.15 else "fragment"
    merge = sfm.get("merge") or {}
    return {"models": out, "registered": total, "merged": merge.get("status") == "ok",
            "models_before_merge": merge.get("models_before"), "models_after_merge": merge.get("models_after"),
            "generated": generated_object(run), "complete": complete_object(run)}


def generated_object(run: Path) -> dict:
    """The run's generated object (drone3d.generate): its state, files and what it was made from."""
    from drone3d import generate

    rec = generate.status(run) or {}
    state = rec.get("status") or "none"
    if state == "running" and not _alive(rec.get("pid")):
        state = "interrupted"  # the process that ran it is gone (console restarted)
    base = f"/runs/{run.name}/export/generated"
    return {"available": generate.available(), "status": state, "keyframe": rec.get("keyframe"),
            "seconds": rec.get("seconds"), "resolution": rec.get("resolution"), "note": generate.NOTE,
            "glb": "generated/object.glb" if state == "ok" and rec.get("glb") else None,
            "input": f"{base}/{rec['input']}" if rec.get("input") else None, "started": rec.get("started"),
            "error": (rec.get("log") or [""])[-1][:300] if state == "failed" else None,
            "placed": (rec.get("aligned") or {}).get("measured_covered") if (rec.get("aligned") or {}).get("placed") else None}  # fmt: skip


COMPLETE_JOBS: dict[str, str] = {}  # run name -> the complete-model job running for it ("complete" | "splats")


def complete_object(run: Path) -> dict:
    """The run's complete model (drone3d.complete): its state, the numbers that say what it is, its files."""
    out = run / "export" / "complete"
    rec = _read(out / "result.json") or {}
    sp = _read(out / "splats_360.json") or {}
    base = f"/runs/{run.name}/export/complete"
    job = COMPLETE_JOBS.get(run.name)
    files = [f for f in ("subject.glb", "subject.obj", "subject.fbx", "subject.stl", "scene.glb", "splats_360.splat") if (out / f).is_file()]
    return {"status": "running" if job == "complete" else rec.get("status") or "none",
            "splats": "running" if job == "splats" else ("ok" if (out / "splats_360.splat").is_file() else "none"),
            "watertight": rec.get("watertight"), "photographed": round(1 - rec["generated_share"], 3) if "generated_share" in rec else None,
            "triangles": rec.get("subject_triangles"), "consensus_of": rec.get("consensus_of"), "generations": rec.get("generations"),
            "splat_views": sp.get("views"), "num_splats": sp.get("num_splats"),
            "thumb": f"{base}/thumb.jpg" if (out / "thumb.jpg").is_file() else None,
            "files": [{"name": f, "url": f"{base}/{f}"} for f in files], "note": rec.get("note")}  # fmt: skip


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _group(name: str, kind: str) -> str:
    """Where a run is listed: the sample-video benchmark, the user's builds and uploads, live sessions, or development."""
    if kind == "live" or name.startswith("live_"):
        return "live"
    if name.startswith("map_"):
        return "benchmark"
    if re.match(r"^(abl_|ded_|q_|eng_|dev_|repro|sample_|cold\d?_)|^(jal_mahal|qutub)(_fast\d*|_rel\d+)?$", name):
        return "development"  # the runs behind the paper's studies, from earlier versions of the pipeline
    return "builds"


def _same_file(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    import hashlib

    def digest(p: Path) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        with p.open("rb") as fh:
            while chunk := fh.read(8 << 20):
                h.update(chunk)
        return h.digest()

    return digest(a) == digest(b)


def _place(part: Path, dest: Path) -> tuple[Path, bool]:
    """Move an uploaded ``part`` to ``dest`` -> ``(path, reused)``.

    A file of the same name is never overwritten -- runs refer to it by path.
    The same bytes uploaded again reuse it; different ones get ``name-2.ext``, ``-3``...
    """
    for k in range(1, 1000):
        cand = dest if k == 1 else dest.with_name(f"{dest.stem}-{k}{dest.suffix}")
        if not cand.exists():
            part.rename(cand)
            return cand, False
        if _same_file(part, cand):
            return cand, True
    raise HTTPException(409, f"too many uploads named {dest.name}")
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


def create_app(
    repo: Path, outputs: Path | None = None, *, engine_port: int = 8770, engine_slots: int = 1
):  # type: ignore[no-untyped-def]
    from fastapi import FastAPI
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
    engine = EngineClient(repo, outputs, port=engine_port, slots=engine_slots)
    probe_cache: dict[str, dict] = {}

    def own_pids() -> set[int]:
        pids = {os.getpid()} | _descendants(os.getpid())  # with a TRELLIS.2 generation it runs
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
        for root, origin in ((uploads, "upload"), (repo / "datasets", "sample")):  # a fresh upload first
            if not root.is_dir():
                continue
            files = sorted(root.iterdir(), key=lambda p: -p.stat().st_mtime) if origin == "upload" else sorted(root.iterdir())
            for p in files:
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
    def upload(file: UploadFile = File(...)) -> dict:  # noqa: B008  (sync: a 4 GB copy runs in a worker thread, not the event loop)
        name = Path(file.filename or "upload").name
        if Path(name).suffix.lower() not in VIDEO_EXT | LOG_EXT:
            raise HTTPException(400, f"unsupported file type: {name}")
        part = uploads / f".{name}.{threading.get_ident()}.part"  # hidden, and not a video: never listed half-written
        try:
            with part.open("wb") as fh:
                shutil.copyfileobj(file.file, fh, 8 << 20)
            if Path(name).suffix.lower() in VIDEO_EXT:  # refuse now what the ingest stage would fail on later
                from drone3d.io.nvdec import probe_stream

                try:
                    info = probe_stream(part, count_frames=False)
                    ok = info.width > 0 and info.height > 0 and info.duration_s > 0
                except Exception:
                    ok = False
                if not ok:
                    raise HTTPException(400, f"{name} is not a readable video (no video stream with a duration)")
            dest, reused = _place(part, uploads / name)
        finally:
            part.unlink(missing_ok=True)
        return {
            "path": str(dest.relative_to(repo)),
            "name": dest.name,
            "size_mb": round(dest.stat().st_size / 1e6, 1),
            "reused": reused,
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
                s["group"] = _group(d.name, s["kind"])
                photos = sorted((d / "export" / "model_0" / "frames" / "photo").glob("*.jpg")) or sorted(
                    d.glob("segments/*/export/model_0/frames/photo/*.jpg"))  # a live session: its segments' models
                s["thumb"] = f"/runs/{d.name}/{photos[len(photos) // 2].relative_to(d).as_posix()}" if photos else None
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
                job = engine.submit(name, config, front=bool(req.get("next")))  # "run next": ahead of a queued batch
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
            if (m.get("generated") or {}).get("mesh"):  # drone3d.generate's object, placed in this model's frame
                m["generated"] = {**m["generated"], "mesh": f"{url}/export/{m['generated']['mesh']}"}
            if m.get("complete"):  # drone3d.complete: the whole subject in its scene, and 360-degree splats
                m["complete"] = {k: (f"{url}/export/{v}" if k in ("scene", "subject", "splat") and v else v)
                                 for k, v in m["complete"].items()}  # fmt: skip
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

    @app.get("/api/runs/{name}/models")
    def run_models(name: str) -> dict:
        """One entry per model of a run: what it covers and how good it is, main pieces first."""
        if not _NAME.match(name) or not (outputs / name).is_dir():
            raise HTTPException(404, "no such run")
        return model_catalog(outputs / name)

    generating: dict[str, threading.Thread] = {}

    @app.post("/api/runs/{name}/generate")
    def generate_run(name: str, req: dict = Body(default={})) -> dict:  # noqa: B008
        """Generate the run's object with TRELLIS.2 in the background (one at a time; ~5 min a keyframe).

        ``{"views": 3}``: from three keyframes across the flight, the complete model their consensus."""
        from drone3d import generate

        run = outputs / name
        if not _NAME.match(name) or not (run / "sfm" / "result.json").is_file():
            raise HTTPException(404, "no finished run of that name")
        if not generate.available():
            raise HTTPException(409, "TRELLIS.2 is not installed on this machine (tools/setup_trellis2.sh)")
        busy = [k for k, t in generating.items() if t.is_alive()]
        if busy:
            raise HTTPException(409, f"already generating {busy[0]}; one object at a time")

        views = max(1, min(5, int(req.get("views") or 1)))

        def work() -> None:
            try:
                generate.generate_views(run, views=views) if views > 1 else generate.generate_object(run)
            except Exception as exc:  # recorded in result.json by generate_object where it can be
                log.warning("generate %s: %s", name, exc)

        generating[name] = threading.Thread(target=work, name=f"generate-{name}", daemon=True)
        generating[name].start()
        return {"name": name, "status": "running", "views": views}

    def _complete_job(name: str, kind: str) -> dict:
        from drone3d import complete as comp

        run = outputs / name
        if not _NAME.match(name) or not (run / "sfm" / "result.json").is_file():
            raise HTTPException(404, "no finished run of that name")
        busy = [k for k, t in generating.items() if t.is_alive()]
        if busy:
            raise HTTPException(409, f"the GPU is busy with {busy[0]}; one generation or completion at a time")
        if kind == "splats" and not (run / "export" / "complete" / "scene.glb").is_file():
            raise HTTPException(409, "build the complete model first")

        def work() -> None:
            COMPLETE_JOBS[name] = kind
            try:
                comp.complete_model(run) if kind == "complete" else comp.splats_360(run)
            except Exception as exc:  # the files stay as they were
                log.warning("%s %s: %s", kind, name, exc)
            finally:
                COMPLETE_JOBS.pop(name, None)

        generating[name] = threading.Thread(target=work, name=f"{kind}-{name}", daemon=True)
        generating[name].start()
        return {"name": name, "status": "running", "job": kind}

    @app.post("/api/runs/{name}/complete")
    def complete_run(name: str) -> dict:
        """Rebuild the complete model (drone3d.complete) from the placed generated object(s) in the background."""
        return _complete_job(name, "complete")

    @app.post("/api/runs/{name}/splats360")
    def splats360_run(name: str) -> dict:
        """Train Gaussian splats that are whole from every heading (drone3d.complete.splats_360; ~5-10 min)."""
        return _complete_job(name, "splats")

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
                if any(part.startswith(".") for part in img.relative_to(rd).parts):
                    continue  # caches and other hidden files are not keyframes
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
        if any(part.startswith(".") for part in Path(path).parts):
            raise HTTPException(404, "no such image")
        # one hidden cache for every run, outside the runs' own folders (stages scan dataset/images)
        cache = outputs / ".thumbs" / f"{Path(path).with_suffix('')}_{w}.jpg"
        if not cache.is_file() or cache.stat().st_mtime < src.stat().st_mtime:
            from PIL import Image

            cache.parent.mkdir(parents=True, exist_ok=True)
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
        path = Path(
            ((_read(run_dir / "ingest" / "result.json") or {}).get("video") or {}).get("path") or ""
        )
        if not path.is_absolute():
            path = repo / path
        if not path.is_file():
            raise HTTPException(404, "the run's video is not on this machine")
        kind = {".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime"}.get(
            path.suffix.lower(), "video/mp4"
        )
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

    @app.post("/api/engine/restart")
    def engine_restart(req: dict = Body(default={})) -> dict:  # noqa: B008
        """Drain, keep the queue, and come back with new code or a new slot count."""
        try:
            return engine.restart(slots=req.get("slots"))
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
