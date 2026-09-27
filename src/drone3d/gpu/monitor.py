"""Sample GPU utilisation, power and memory while a stage runs (NVML).

Device-level counters include every process on the GPU, so each sample also
records how much memory *other* processes hold; a report can then say whether
a timing was taken on a shared GPU instead of silently mixing the two.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["GpuMonitor", "GpuSummary"]

log = get_logger(__name__)


@dataclass
class GpuSummary:
    samples: int = 0
    seconds: float = 0.0
    util_mean: float | None = None
    util_p90: float | None = None
    power_mean_w: float | None = None
    power_limit_w: float | None = None
    memory_peak_gb: float | None = None
    own_memory_peak_gb: float | None = None
    foreign_processes_max: int = 0
    decoder_util_mean: float | None = None
    host_rss_peak_gb: float | None = None  # this process + its children (spirula, ffmpeg)
    timeline: list[tuple[float, float, float]] = field(default_factory=list)  # (t, util %, W)

    def to_dict(self, with_timeline: bool = False) -> dict:
        d = {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
            if k != "timeline"
        }
        if with_timeline:
            d["timeline"] = [[round(t, 2), u, round(w, 1)] for t, u, w in self.timeline]
        d["shared_gpu"] = self.foreign_processes_max > 0
        return d


class GpuMonitor:
    """Context manager sampling NVML every ``interval`` seconds on a thread."""

    def __init__(self, index: int = 0, interval: float = 0.5) -> None:
        self.index = index
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[tuple[float, float, float, float, float, int, float, float]] = []
        self._limit: float | None = None
        self._ok = False

    def __enter__(self) -> GpuMonitor:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            self._ok = True
        except Exception as exc:  # NVML missing or no GPU: monitoring is optional
            log.debug("GPU monitor disabled: %s", exc)
            return self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        nv, h, me = self._nvml, self._handle, os.getpid()
        children = set()
        while not self._stop.is_set():
            try:
                util = nv.nvmlDeviceGetUtilizationRates(h)
                power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                mem = nv.nvmlDeviceGetMemoryInfo(h).used / 1e9
                dec = float(nv.nvmlDeviceGetDecoderUtilization(h)[0])
                procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
                # Our own work may run in child processes (spirula, ffmpeg).
                children = _descendants(me) | {me}
                own = sum((p.usedGpuMemory or 0) for p in procs if p.pid in children) / 1e9
                foreign = sum(1 for p in procs if p.pid not in children)
                rss = sum(_rss_bytes(pid) for pid in children) / 1e9
                self._rows.append(
                    (time.perf_counter() - self._t0, float(util.gpu), power, mem, own, foreign, dec, rss)
                )
            except Exception:  # a transient NVML error must not kill the stage
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> GpuSummary:
        if not self._rows:
            return GpuSummary()
        a = np.array(self._rows, dtype=np.float64)
        return GpuSummary(
            samples=len(a),
            seconds=float(a[-1, 0]),
            util_mean=float(a[:, 1].mean()),
            util_p90=float(np.percentile(a[:, 1], 90)),
            power_mean_w=float(a[:, 2].mean()),
            power_limit_w=self._limit,
            memory_peak_gb=float(a[:, 3].max()),
            own_memory_peak_gb=float(a[:, 4].max()),
            foreign_processes_max=int(a[:, 5].max()),
            decoder_util_mean=float(a[:, 6].mean()),
            host_rss_peak_gb=float(a[:, 7].max()),
            timeline=[(float(t), float(u), float(w)) for t, u, w in a[:, :3]],
        )


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _descendants(pid: int) -> set[int]:
    """PIDs of every descendant of ``pid`` (Linux /proc)."""
    out: set[int] = set()
    try:
        children: dict[int, list[int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(entry.name))
        stack = [pid]
        while stack:
            for child in children.get(stack.pop(), []):
                if child not in out:
                    out.add(child)
                    stack.append(child)
    except OSError:
        pass
    return out
