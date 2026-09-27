"""The resident engine: warm models, a job queue, live-stream sessions and an HTTP control API.

``drone3d engine`` runs one of these per GPU. Reconstructions execute in this
process, one at a time, drawing their networks from a shared
:class:`~drone3d.engine.models.ModelCache`, so a video starts without loading
anything. Each run writes the same files as ``drone3d run`` (``logs/run.log``,
stage results, ``ui_job.json``), which is all the web console reads.

Before a run, memory-hungry settings are fitted to the GPU memory that is free
*at that moment* (:func:`fit_to_gpu`): the card may be shared, and a run that
takes what another user holds would crash one of the two.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from drone3d.engine import models
from drone3d.engine.live import LiveSession
from drone3d.gpu.nvml import free_bytes, snapshot
from drone3d.logging_utils import _DATE_FORMAT, _LOG_FORMAT, get_logger

__all__ = ["Engine", "create_engine_app", "fit_to_gpu", "models_for"]

log = get_logger(__name__)
GiB = 2**30
POISONED_EXIT = 3  # exit code after a sticky CUDA error: the supervisor restarts the engine


def cuda_healthy() -> str | None:
    """None if the CUDA context still works, else the error (a sticky fault poisons the process)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.ones(1, device="cuda").add_(1).item()
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    return None


def models_for(config: dict[str, Any]) -> list[str]:
    """Keys of the warm models a run with this (plain-dict) config will use."""
    stages = set(config.get("stages") or [])
    kf, sfm, dense = (
        config.get("keyframes") or {},
        config.get("sfm") or {},
        config.get("dense") or {},
    )
    out = []
    if "keyframes" in stages:
        out.append(kf.get("flow_model", "raft_large"))
    if ("sfm" in stages and sfm.get("backend") == "flow") or (
        "dense" in stages and dense.get("backend", "flow") == "flow"
    ):
        out.append("raft_large")
    if (
        "dense" in stages
        and dense.get("backend", "flow") == "flow"
        and dense.get("mono_model", models.DA_V2_LARGE) == models.DA_V2_LARGE
    ):
        out.append("depth_anything_v2_large")
    if "depth" in stages and (config.get("depth") or {}).get("backend", "marigold") == "marigold":
        out.append("marigold_v2")
    return list(dict.fromkeys(out))


def fit_to_gpu(config: dict[str, Any], free_gb: float | None) -> list[str]:
    """Scale memory-hungry settings of ``config`` (in place) to ``free_gb``; returns what changed."""
    if free_gb is None:
        return []
    changes = []
    dense = config.setdefault("dense", {})
    tsdf = float(dense.get("tsdf_memory_gb", 8.0))
    cap = max(2.0, round(0.45 * free_gb, 1))
    if tsdf > cap:
        dense["tsdf_memory_gb"] = cap
        changes.append(f"dense.tsdf_memory_gb {tsdf:g} -> {cap:g} ({free_gb:.1f} GiB free)")
    kf = config.setdefault("keyframes", {})
    batch = int(kf.get("flow_batch", 32))
    fit = 32 if free_gb >= 16 else 16 if free_gb >= 9 else 8
    if batch > fit:
        kf["flow_batch"] = fit
        changes.append(f"keyframes.flow_batch {batch} -> {fit} ({free_gb:.1f} GiB free)")
    ex = config.setdefault("export", {})
    size = int(ex.get("texture_size", 4096))
    if size > 4096 and free_gb < 12:
        ex["texture_size"] = 4096
        changes.append(f"export.texture_size {size} -> 4096 ({free_gb:.1f} GiB free)")
    return changes


@dataclass
class Job:
    name: str
    run_dir: str
    config: dict[str, Any]
    kind: str = "run"  # run | segment
    live: str | None = None
    status: str = "queued"  # queued | running | done | failed | stopped
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    seconds: float | None = None
    adjustments: list[str] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    warm: list[str] = field(default_factory=list)  # models already loaded when it started
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        d = {
            f.name: copy.deepcopy(getattr(self, f.name))
            for f in fields(self)
            if f.name not in ("config", "cancel")
        }
        d["video"] = (self.config.get("ingest") or {}).get("video")
        return d


class Engine:
    """Queue + warm models; :meth:`serve` exposes it over HTTP."""

    def __init__(self, repo: Path, outputs: Path, *, device: str = "cuda", reserve_gb: float = 2.0,
                 release_after_job: bool = True) -> None:  # fmt: skip
        self.repo, self.outputs = Path(repo), Path(outputs)
        self.cache = models.ModelCache(device=device, reserve_gb=reserve_gb)
        models.activate(self.cache)
        self.release_after_job = release_after_job
        self.started = time.time()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self.queue: collections.deque[Job] = collections.deque()
        self.current: Job | None = None
        self.history: collections.deque[Job] = collections.deque(maxlen=50)
        self.live: dict[str, LiveSession] = {}
        self._stop = threading.Event()
        self.poisoned: str | None = None
        self._restore_queue()
        self._thread = threading.Thread(target=self._loop, name="engine-jobs", daemon=True)
        self._thread.start()

    # ------------------------------------------------------- restart safety
    @property
    def _queue_file(self) -> Path:
        return self.outputs / ".engine_queue.json"

    def _save_queue(self) -> None:
        with self._lock:
            rows = [{"name": j.name, "run_dir": j.run_dir, "config": j.config, "kind": j.kind, "live": j.live}
                    for j in self.queue]  # fmt: skip
        self.outputs.mkdir(parents=True, exist_ok=True)
        self._queue_file.write_text(json.dumps(rows))

    def _restore_queue(self) -> None:
        """Jobs a previous engine left queued when its CUDA context died."""
        try:
            rows = json.loads(self._queue_file.read_text())
        except (OSError, ValueError):
            return
        self._queue_file.unlink(missing_ok=True)
        for r in rows:
            try:
                self.submit(
                    r["name"],
                    r["config"],
                    run_dir=r["run_dir"],
                    kind=r.get("kind", "run"),
                    live=r.get("live"),
                )
            except Exception as exc:
                log.warning("could not restore queued job %s: %s", r.get("name"), exc)

    # ---------------------------------------------------------------- jobs
    def submit(self, name: str, config: dict[str, Any], *, run_dir: str | Path | None = None, kind: str = "run",
               live: str | None = None, front: bool = False) -> Job:  # fmt: skip
        from drone3d.config import PipelineConfig, _build

        _build(PipelineConfig, copy.deepcopy(config)).validate()
        run = Path(run_dir) if run_dir else self.outputs / name
        with self._lock:
            if any(j.name == name for j in self.queue) or (
                self.current and self.current.name == name
            ):
                raise ValueError(f"run {name!r} is already queued or running")
            job = Job(
                name=name, run_dir=str(run), config=copy.deepcopy(config), kind=kind, live=live
            )
            run.mkdir(parents=True, exist_ok=True)
            import yaml

            (run / "ui_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            self._state(job)
            (self.queue.appendleft if front else self.queue.append)(job)
        self._wake.set()
        return job

    def cancel(self, name: str) -> bool:
        with self._lock:
            for job in list(self.queue):
                if job.name == name:
                    self.queue.remove(job)
                    job.status, job.finished = "stopped", time.time()
                    self._state(job)
                    self.history.appendleft(job)
                    return True
            if self.current and self.current.name == name:
                self.current.cancel.set()  # stops before the next stage
                self._state(self.current, status="stopping")
                return True
        return False

    def _state(self, job: Job, **extra: Any) -> None:
        path = Path(job.run_dir) / "ui_job.json"
        state = {"status": job.status, "submitted": job.submitted, "started": job.started, "finished": job.finished,
                 "pid": os.getpid(), "engine": True, "kind": job.kind, "live": job.live, "warm": job.warm,
                 "adjustments": job.adjustments, "exit_code": None if job.status in ("queued", "running") else
                 (0 if job.status == "done" else 1), **extra}  # fmt: skip
        with contextlib.suppress(OSError):
            path.write_text(json.dumps(state, indent=1))

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job = self.queue.popleft() if self.queue else None
                self.current = job
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            try:
                self._execute(job)
            except Exception as exc:  # an engine bug must not take the queue down
                log.exception("job %s crashed the engine loop", job.name)
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                job.seconds = round(job.finished - (job.started or job.finished), 2)
                self._state(job)
                with self._lock:
                    self.current = None
                    self.history.appendleft(job)
                for session in list(self.live.values()):
                    session.job_finished(job)
            fault = cuda_healthy()
            if fault:  # every later job would fail in this process: hand over to a fresh one
                self.poisoned = f"CUDA context lost after {job.name}: {fault}"
                log.error("engine: %s; saving the queue and exiting for a restart", self.poisoned)
                self._save_queue()
                self._write_warm_list()
                os._exit(POISONED_EXIT)

    def _execute(self, job: Job) -> None:
        from drone3d.config import PipelineConfig, _build
        from drone3d.pipeline import Pipeline

        free = free_bytes()
        job.adjustments = fit_to_gpu(job.config, None if free is None else free / GiB)
        cfg = _build(PipelineConfig, copy.deepcopy(job.config))
        cfg.validate()
        run_dir = Path(job.run_dir)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(run_dir / "logs" / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        level = root.level
        root.setLevel(cfg.log_level)  # the run's own level, as ``drone3d run`` would set it
        job.warm = [m["key"] for m in self.cache.status() if m["status"] == "loaded"]
        job.status, job.started = "running", time.time()
        self._state(job)
        try:
            for change in job.adjustments:
                log.info("engine: fitted to free GPU memory: %s", change)
            log.info("engine: warm models at start: %s", ", ".join(job.warm) or "none")
            pipeline = Pipeline(cfg, run_dir)
            pipeline.cancel = job.cancel
            with self.cache.job():
                result = pipeline.run()
            job.stages = [{"name": s.name, "status": s.status, "seconds": round(s.duration_s, 2), "message": s.message}
                          for s in result.stages]  # fmt: skip
            if job.cancel.is_set():
                job.status = "stopped"
            else:
                job.status = "done" if result.ok else "failed"
        except Exception as exc:
            log.exception("run %s failed", job.name)
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            root.removeHandler(handler)
            root.setLevel(level)
            handler.close()
            if self.release_after_job:  # activations and TSDF blocks, not the warm weights
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --------------------------------------------------------------- models
    def warm(self, keys: list[str]) -> dict[str, str]:
        try:
            return self._warm(keys)
        finally:
            self._write_warm_list()

    def _warm(self, keys: list[str]) -> dict[str, str]:
        out = {}
        for key in keys:
            try:
                self.cache.load(key)
                out[key] = "loaded"
            except models.InsufficientMemory as exc:
                out[key] = f"refused: {exc}"
            except Exception as exc:
                out[key] = f"error: {type(exc).__name__}: {exc}"
        return out

    def _write_warm_list(self) -> None:
        loaded = [m["key"] for m in self.cache.status() if m["status"] == "loaded"]
        with contextlib.suppress(OSError):
            (self.outputs / ".engine_warm.json").write_text(json.dumps(loaded))

    # ----------------------------------------------------------------- live
    def start_live(self, name: str, source: str, config: dict[str, Any], *, segment_s: float = 30.0,
                   simulate: bool = False, telemetry: str | None = None) -> LiveSession:  # fmt: skip
        with self._lock:
            if name in self.live and self.live[name].running:
                raise ValueError(f"live session {name!r} is already running")
            session = LiveSession(self, name, source, self.outputs / name, config, segment_s=segment_s,
                                  simulate=simulate, telemetry=telemetry)  # fmt: skip
            self.live[name] = session
        session.start()
        return session

    def stop_live(self, name: str) -> bool:
        session = self.live.get(name)
        if session is None:
            return False
        session.stop()
        return True

    # --------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        import torch

        mem = {}
        if torch.cuda.is_available():
            mem = {"allocated_mb": round(torch.cuda.memory_allocated() / 2**20),
                   "reserved_mb": round(torch.cuda.memory_reserved() / 2**20)}  # fmt: skip
        with self._lock:
            return {
                "pid": os.getpid(),
                "uptime_s": round(time.time() - self.started, 1),
                "device": self.cache.device,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "memory": mem,
                "models": self.cache.status(),
                "current": self.current.public() if self.current else None,
                "queue": [j.public() for j in self.queue],
                "history": [j.public() for j in list(self.history)[:20]],
                "live": {n: s.status() for n, s in self.live.items()},
                "gpu": snapshot({os.getpid()}),
                "poisoned": self.poisoned,
            }

    def shutdown(self) -> None:
        for session in list(self.live.values()):
            session.stop()
        self._stop.set()
        self._wake.set()
        if self.current:
            self.current.cancel.set()
        self._thread.join(timeout=5)
        self.cache.unload_all()


def _check(fn) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    try:
        detail = fn()
        return {"ok": True, "detail": detail, "ms": round(1e3 * (time.perf_counter() - t0), 1)}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


def capabilities(repo: Path) -> dict[str, Any]:
    """What this machine can do, checked for real: codecs, libraries, binaries and weights."""
    import subprocess

    def torch_cuda() -> str:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        p = torch.cuda.get_device_properties(0)
        return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {p.name}, {p.total_memory / GiB:.0f} GiB"

    def nvjpeg() -> str:
        import cv2
        import numpy as np
        import torch
        from torchvision.io import decode_jpeg

        ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
        decode_jpeg(torch.from_numpy(buf.ravel()), device="cuda")
        return "GPU JPEG decode (torchvision / nvJPEG)"

    def ffmpeg(kind: str) -> str:
        from drone3d.io.nvdec import ffmpeg_bin

        exe = ffmpeg_bin()
        if kind == "nvdec":
            out = subprocess.run([exe, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10).stdout
            if "cuda" not in out:
                raise RuntimeError("ffmpeg has no cuda hwaccel")
            return f"{Path(exe).parent.parent.name or 'ffmpeg'}: -hwaccel cuda"
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
        if "h264_nvenc" not in out:
            raise RuntimeError("ffmpeg has no h264_nvenc")
        return "h264_nvenc, hevc_nvenc"

    def open3d() -> str:
        import open3d as o3d

        if not o3d.core.cuda.is_available():
            raise RuntimeError(f"Open3D {o3d.__version__} without CUDA")
        return f"Open3D {o3d.__version__}, CUDA TSDF"

    def pycolmap() -> str:
        import pycolmap as pc

        return f"pycolmap {pc.__version__}, global mapper {'yes' if hasattr(pc, 'global_mapping') else 'no'}"

    def spirula() -> str:
        from drone3d.splat.spirula import spirula_binary

        return str(spirula_binary().relative_to(repo)) if spirula_binary().is_relative_to(repo) else str(spirula_binary())

    def meshconv() -> str:
        exe = repo / ".tools" / "bin" / "meshconv"
        if not exe.is_file():
            raise FileNotFoundError("build with tools/build_meshconv.sh")
        return "assimp FBX writer (.tools/bin/meshconv)"

    def weights(path: Path, what: str) -> str:
        if not path.exists():
            raise FileNotFoundError(str(path))
        return what

    hf = Path(os.environ.get("HF_HUB_CACHE", "/store/huggingface"))
    return {
        "cuda": _check(torch_cuda),
        "nvdec": _check(lambda: ffmpeg("nvdec")),
        "nvenc": _check(lambda: ffmpeg("nvenc")),
        "nvjpeg": _check(nvjpeg),
        "open3d": _check(open3d),
        "pycolmap": _check(pycolmap),
        "spirula": _check(spirula),
        "meshconv": _check(meshconv),
        "depth_anything": _check(lambda: weights(hf / "models--depth-anything--Depth-Anything-V2-Large-hf",
                                                 "Depth Anything V2 Large weights")),  # fmt: skip
        "marigold": _check(lambda: weights(Path("/store/huggingface/marigold-v2"), "Marigold v2 + Qwen weights")),
    }


def create_engine_app(engine: Engine):  # type: ignore[no-untyped-def]
    """The engine's HTTP control API (FastAPI)."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="drone3d engine")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "pid": os.getpid(), "uptime_s": round(time.time() - engine.started, 1)}

    @app.get("/status")
    def status() -> dict:
        return engine.status()

    caps: dict[str, Any] = {}

    @app.get("/capabilities")
    def get_capabilities() -> dict:
        if not caps:
            caps.update(capabilities(engine.repo))
        return caps

    @app.post("/models/{key}/load")
    def load(key: str) -> dict:
        if key not in models.SPECS:
            raise HTTPException(404, f"unknown model {key!r}")
        return {key: engine.warm([key])[key]}

    @app.post("/models/{key}/unload")
    def unload(key: str) -> dict:
        if key not in models.SPECS:
            raise HTTPException(404, f"unknown model {key!r}")
        out = {key: engine.cache.unload(key)}
        engine._write_warm_list()
        return out

    @app.post("/models/warm")
    def warm(body: dict) -> dict:
        keys = body.get("keys") or (
            models_for(body["config"]) if body.get("config") else list(models.SPECS)[:1]
        )
        return engine.warm([k for k in keys if k in models.SPECS])

    @app.post("/models/unload")
    def unload_all() -> dict:
        return {
            m["key"]: engine.cache.unload(m["key"])
            for m in engine.cache.status()
            if m["status"] == "loaded"
        }

    @app.post("/jobs")
    def submit(body: dict) -> dict:
        try:
            job = engine.submit(body["name"], body["config"], run_dir=body.get("run_dir"))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # ConfigError and friends
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        return job.public()

    @app.post("/jobs/{name}/cancel")
    def cancel(name: str) -> dict:
        if not engine.cancel(name):
            raise HTTPException(404, f"no queued or running job {name!r}")
        return {"ok": True}

    @app.post("/live")
    def live(body: dict) -> dict:
        try:
            session = engine.start_live(body["name"], body["source"], body["config"],
                                        segment_s=float(body.get("segment_s", 30)), simulate=bool(body.get("simulate")),
                                        telemetry=body.get("telemetry"))  # fmt: skip
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return session.status()

    @app.post("/live/{name}/stop")
    def live_stop(name: str) -> dict:
        if not engine.stop_live(name):
            raise HTTPException(404, f"no live session {name!r}")
        return {"ok": True}

    @app.post("/shutdown")
    def shutdown(body: dict | None = None) -> dict:
        if engine.current and not (body or {}).get("force"):
            raise HTTPException(409, f"busy with {engine.current.name}; pass force to cancel it")
        threading.Thread(
            target=lambda: (time.sleep(0.3), engine.shutdown(), os._exit(0)), daemon=True
        ).start()
        return {"ok": True}

    return app


def serve(repo: Path, outputs: Path, *, host: str = "127.0.0.1", port: int = 8765, warm: list[str] | None = None,
          reserve_gb: float = 2.0) -> None:  # fmt: skip
    """Run an engine until killed; ``outputs/.engine.json`` tells the console where it is."""
    import uvicorn

    engine = Engine(repo, outputs, reserve_gb=reserve_gb)
    if warm is None:  # a restart: bring back what the previous engine held
        try:
            warm = json.loads((outputs / ".engine_warm.json").read_text())
        except (OSError, ValueError):
            warm = []
    if warm:
        threading.Thread(target=engine.warm, args=(warm,), daemon=True).start()
    info = outputs / ".engine.json"
    outputs.mkdir(parents=True, exist_ok=True)
    info.write_text(
        json.dumps({"pid": os.getpid(), "host": host, "port": port, "started": engine.started})
    )
    try:
        uvicorn.run(create_engine_app(engine), host=host, port=port, log_level="warning")
    finally:
        try:
            if json.loads(info.read_text()).get("pid") == os.getpid():
                info.unlink()
        except (OSError, ValueError):
            pass
        engine.shutdown()
