"""Live footage: record a stream in segments and reconstruct each one while the next is recorded.

ffmpeg reads the source (RTSP / RTMP / SRT / UDP / HLS / HTTP, a V4L2 camera,
or a file replayed at its own frame rate to rehearse a flight) and cuts it
into ``segment_s`` pieces at keyframes without re-encoding. Every closed
segment becomes an ordinary engine run under ``<session>/segments/`` with the
warm models, queued ahead of other work, so a model of the first minute of
flight exists while the drone is still in the air. With a flight log, every
segment is georeferenced into one shared ENU frame (the origin is pinned to
the log's first fix), so the segments' models line up in one scene.

``<session>/live.json`` is the session's state; the console polls it.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drone3d.logging_utils import get_logger

if TYPE_CHECKING:
    from drone3d.engine.service import Engine, Job

__all__ = ["LiveSession", "recorder_command"]

log = get_logger(__name__)


def recorder_command(
    ffmpeg: str, source: str, seg_dir: Path, segment_s: float, *, simulate: bool = False
) -> list[str]:
    """ffmpeg arguments that record ``source`` into ``seg_dir/seg_%04d.mkv`` plus ``segments.csv``."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin", "-y"]
    raw = source.startswith("/dev/video")
    if simulate:
        cmd += ["-re"]  # replay a file at its frame rate, as a drone link would deliver it
    if source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    if raw:
        cmd += ["-f", "v4l2"]
    if source.split(":", 1)[0] in ("rtsp", "rtmp", "srt", "udp", "http", "https", "tcp"):
        cmd += ["-rw_timeout", "15000000"]  # give up on a dead link after 15 s instead of hanging
    cmd += ["-i", source, "-map", "0:v:0", "-an"]
    if raw:  # a camera delivers raw frames: encode on NVENC with a keyframe every second
        cmd += ["-c:v", "h264_nvenc", "-preset", "p2", "-g", "30", "-b:v", "40M"]
    else:
        cmd += ["-c", "copy"]  # cut at the stream's own keyframes, no re-encode
    cmd += ["-f", "segment", "-segment_time", f"{segment_s:g}", "-segment_format", "matroska",
            "-reset_timestamps", "1", "-segment_list", str(seg_dir / "segments.csv"), "-segment_list_type", "csv",
            str(seg_dir / "seg_%04d.mkv")]  # fmt: skip
    return cmd


def _first_fix(telemetry: str) -> tuple[float, float, float | None] | None:
    from drone3d.io.telemetry import load_telemetry

    for s in load_telemetry(telemetry):
        if s.lat is not None and s.lon is not None:
            return s.lat, s.lon, s.alt_m
    return None


class LiveSession:
    """One stream: a recorder process, a watcher thread and the segment runs it queues."""

    def __init__(self, engine: Engine, name: str, source: str, run_dir: Path, config: dict[str, Any], *,
                 segment_s: float = 30.0, simulate: bool = False, telemetry: str | None = None) -> None:  # fmt: skip
        if segment_s < 5:
            raise ValueError(
                "segment_s must be at least 5 s (a pass needs a few seconds of motion)"
            )
        self.engine, self.name, self.source, self.run_dir = engine, name, source, Path(run_dir)
        self.config, self.segment_s, self.simulate, self.telemetry = (
            copy.deepcopy(config),
            segment_s,
            simulate,
            telemetry,
        )
        self.seg_dir = self.run_dir / "segments"
        self.segments: list[dict[str, Any]] = []
        self.proc: subprocess.Popen | None = None
        self.started: float | None = None
        self.stopped: float | None = None
        self.error: str | None = None
        self.origin: tuple[float, float, float | None] | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        import yaml

        from drone3d.io.nvdec import ffmpeg_bin

        self.seg_dir.mkdir(parents=True, exist_ok=True)
        (self.seg_dir / "segments.csv").unlink(missing_ok=True)
        (self.run_dir / "ui_config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        if self.telemetry:
            self.origin = _first_fix(self.telemetry)
        cmd = recorder_command(
            ffmpeg_bin(), self.source, self.seg_dir, self.segment_s, simulate=self.simulate
        )
        log_path = self.run_dir / "logs" / "recorder.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as out:
            self.proc = subprocess.Popen(
                cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )
        self.started = time.time()
        log.info(
            "live %s: recording %s in %gs segments (pid %d)",
            self.name,
            self.source,
            self.segment_s,
            self.proc.pid,
        )
        self._write()
        threading.Thread(target=self._watch, name=f"live-{self.name}", daemon=True).start()

    def stop(self) -> None:
        """Stop recording; the segment in progress is closed and still reconstructed."""
        self._stop.set()
        if self.running:
            with contextlib.suppress(OSError):  # ffmpeg finishes the open segment on SIGINT
                os.killpg(self.proc.pid, signal.SIGINT)

    # ------------------------------------------------------------ internals
    def _watch(self) -> None:
        listed = self.seg_dir / "segments.csv"
        seen = 0
        while True:
            exited = self.proc.poll() is not None
            rows = []
            if listed.is_file():
                with listed.open(newline="") as fh:
                    rows = [r for r in csv.reader(fh) if len(r) >= 3]
            for row in rows[seen:]:
                self._queue_segment(len(self.segments), row[0], float(row[1]), float(row[2]))
            seen = len(rows)
            if exited:
                code = self.proc.returncode
                if (
                    code not in (0, 255, -2) and not self._stop.is_set()
                ):  # 255 / -2: stopped with SIGINT
                    self.error = f"recorder exited with code {code} (see logs/recorder.log)"
                self.stopped = time.time()
                self._write()
                return
            time.sleep(0.5)

    def _queue_segment(self, index: int, file: str, start: float, end: float) -> None:
        path = self.seg_dir / file
        name = f"{self.name}__{path.stem}"
        cfg = copy.deepcopy(self.config)
        cfg.setdefault("ingest", {})["video"] = str(path)
        cfg["stages"] = [
            s for s in cfg.get("stages", []) if s != "report"
        ]  # latency first; the session has one view
        offset = self._offset() + start
        if self.telemetry:
            cfg["ingest"].update(telemetry=self.telemetry, telemetry_offset_s=round(offset, 3))
            if self.origin:
                geo = cfg.setdefault("geo", {})
                geo.update(origin_lat=self.origin[0], origin_lon=self.origin[1])
                if self.origin[2] is not None:
                    geo["origin_alt"] = self.origin[2]
        entry = {"index": index, "file": file, "start_s": round(offset, 3), "duration_s": round(end - start, 3),
                 "closed_at": time.time(), "run": name, "run_dir": str(self.run_dir / "segments" / path.stem),
                 "status": "queued"}  # fmt: skip
        with self._lock:
            self.segments.append(entry)
        try:
            self.engine.submit(
                name, cfg, run_dir=entry["run_dir"], kind="segment", live=self.name, front=True
            )
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        log.info("live %s: segment %d (%.1f s) queued", self.name, index, end - start)
        self._write()

    def _offset(self) -> float:
        """Time of the recording's first frame on the flight clock: segment times restart at every run."""
        return float((self.config.get("ingest") or {}).get("telemetry_offset_s", 0.0))

    def job_finished(self, job: Job) -> None:
        if job.live != self.name:
            return
        with self._lock:
            for seg in self.segments:
                if seg["run"] == job.name:
                    seg.update(status=job.status, processing_s=job.seconds, ready_at=job.finished,
                               latency_s=round((job.finished or time.time()) - seg["closed_at"], 2),
                               realtime_factor=round((job.seconds or 0) / max(seg["duration_s"], 1e-3), 3))  # fmt: skip
        self._write()

    def status(self) -> dict[str, Any]:
        with self._lock:
            done = [s for s in self.segments if s.get("status") == "done"]
            return {
                "name": self.name, "source": self.source, "simulate": self.simulate, "segment_s": self.segment_s,
                "recording": self.running, "started": self.started, "stopped": self.stopped, "error": self.error,
                "pid": self.proc.pid if self.proc else None, "telemetry": self.telemetry,
                "origin": list(self.origin) if self.origin else None, "segments": list(self.segments),
                "recorded_s": round(sum(s["duration_s"] for s in self.segments), 2),
                "mean_realtime_factor": round(sum(s["realtime_factor"] for s in done) / len(done), 3) if done else None,
            }  # fmt: skip

    def _write(self) -> None:
        state = self.status()
        try:
            (self.run_dir / "live.json").write_text(json.dumps(state, indent=1))
            pending = any(s.get("status") in ("queued", "running") for s in state["segments"])
            status = (
                "running"
                if (state["recording"] or pending)
                else ("failed" if self.error else "done")
            )
            (self.run_dir / "ui_job.json").write_text(json.dumps({
                "status": status, "kind": "live", "engine": True, "pid": os.getpid(), "submitted": self.started,
                "started": self.started, "finished": None if status == "running" else time.time(),
                "exit_code": None if status == "running" else (1 if self.error else 0)}, indent=1))  # fmt: skip
        except OSError:
            pass
