"""One-call NVML snapshots of every GPU: load, memory, power, clocks, codecs and who holds memory.

The web console polls this once a second and the engine consults it before
loading a model, so both must stay cheap (no subprocess, no torch import) and
must never raise: a machine without NVML reports ``{"available": False}``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

__all__ = ["free_bytes", "snapshot"]

_lock = threading.Lock()
_nv: Any = None


def _nvml() -> Any:
    global _nv
    with _lock:
        if _nv is None:
            import pynvml

            pynvml.nvmlInit()
            _nv = pynvml
    return _nv


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return ""
    parts = [p.decode(errors="replace") for p in raw if p]
    if not parts:
        return ""
    exe = Path(parts[0]).name
    if exe.startswith("python") and len(parts) > 1:  # "python -m x" / "python x.py": name the script
        rest = parts[2] if parts[1] == "-m" and len(parts) > 2 else parts[1]
        return f"{exe} {Path(rest).name}"
    return exe


def _safe(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception:  # an unsupported counter on this GPU is simply absent
        return None


def snapshot(own_pids: set[int] | None = None) -> dict[str, Any]:
    """``{"available", "driver", "cuda", "gpus": [...]}``; ``own_pids`` marks our processes."""
    try:
        nv = _nvml()
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    own = own_pids or {os.getpid()}
    driver = _safe(nv.nvmlSystemGetDriverVersion)
    cuda = _safe(nv.nvmlSystemGetCudaDriverVersion_v2)
    gpus = []
    for i in range(_safe(nv.nvmlDeviceGetCount) or 0):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        mem = _safe(nv.nvmlDeviceGetMemoryInfo, h)
        util = _safe(nv.nvmlDeviceGetUtilizationRates, h)
        power = _safe(nv.nvmlDeviceGetPowerUsage, h)
        limit = _safe(nv.nvmlDeviceGetEnforcedPowerLimit, h)
        enc = _safe(nv.nvmlDeviceGetEncoderUtilization, h)
        dec = _safe(nv.nvmlDeviceGetDecoderUtilization, h)
        procs = []
        for p in _safe(nv.nvmlDeviceGetComputeRunningProcesses, h) or []:
            procs.append({"pid": p.pid, "used_mb": round((p.usedGpuMemory or 0) / 2**20), "name": _cmdline(p.pid),
                          "own": p.pid in own})  # fmt: skip
        gpus.append({
            "index": i,
            "name": _safe(nv.nvmlDeviceGetName, h),
            "util": util.gpu if util else None,
            "mem_util": util.memory if util else None,
            "mem_total_mb": round(mem.total / 2**20) if mem else None,
            "mem_used_mb": round(mem.used / 2**20) if mem else None,
            "mem_free_mb": round(mem.free / 2**20) if mem else None,
            "power_w": round(power / 1000, 1) if power is not None else None,
            "power_limit_w": round(limit / 1000) if limit is not None else None,
            "temp_c": _safe(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": _safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
            "mem_clock_mhz": _safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM),
            "encoder_util": enc[0] if enc else None,
            "decoder_util": dec[0] if dec else None,
            "processes": sorted(procs, key=lambda p: -p["used_mb"]),
        })  # fmt: skip
    cuda_s = f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None
    return {"available": True, "driver": driver, "cuda": cuda_s, "gpus": gpus}


def free_bytes(index: int = 0) -> int | None:
    """Free memory on GPU ``index`` right now (all processes counted), or None without NVML."""
    try:
        nv = _nvml()
        return int(nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(index)).free)
    except Exception:
        return None
