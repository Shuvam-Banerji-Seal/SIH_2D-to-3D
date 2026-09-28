"""Warm models: a process-wide cache the pipeline stages draw their networks from.

Outside the engine (``drone3d run``) nothing is cached: every accessor builds
its model exactly as before and the stage frees it when done. Inside the
engine (:func:`activate`) networks stay on the GPU between runs, so a video
starts on a card that already holds RAFT and Depth Anything, and RAFT's CUDA
graphs for a resolution are captured once instead of three times per run.

Loading is guarded: a model is loaded only if the GPU has its expected
footprint plus a reserve free *now* (other users' processes included), and it
is never unloaded while a run is using it.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from drone3d.gpu.nvml import free_bytes
from drone3d.logging_utils import get_logger

__all__ = [
    "SPECS",
    "InsufficientMemory",
    "ModelCache",
    "activate",
    "active",
    "marigold",
    "mono",
    "raft",
]

log = get_logger(__name__)
GiB = 2**30
DA_V2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
MOGE_3_VITL = "Ruicheng/moge-3-vitl"


class InsufficientMemory(RuntimeError):
    """The GPU does not have room for a model right now."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    role: str
    need_gb: float  # weights + the working set of one batch, measured on the A100
    stages: tuple[str, ...]
    source: str
    license: str


SPECS: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec("raft_large", "RAFT large", "optical flow: keyframe overlap, SfM tracks, dense flow triangulation", 4.0,
              ("keyframes", "sfm", "dense"), "torchvision Raft_Large_Weights.DEFAULT", "BSD-3-Clause"),
    ModelSpec("raft_small", "RAFT small", "optical flow, about 2x faster than large (keyframes.flow_model)", 2.0,
              ("keyframes",), "torchvision Raft_Small_Weights.DEFAULT", "BSD-3-Clause"),
    ModelSpec("depth_anything_v2_large", "Depth Anything V2 Large", "monocular depth: fills what flow cannot triangulate",
              3.5, ("dense",), DA_V2_LARGE, "CC-BY-NC-4.0"),
    ModelSpec("moge_3_vitl", "MoGe-3 ViT-L", "monocular geometry: the fill's prior when dense.mono_model names it",
              6.6, ("dense",), MOGE_3_VITL, "MIT"),
    ModelSpec("marigold_v2", "Marigold v2 (NF4)", "diffusion depth: supervises the Gaussian splats (accurate profile)",
              15.0, ("depth",), "prs-eth Marigold v2 on Qwen-Image-Edit, 4-bit", "Apache-2.0 / Qwen license"),
]}  # fmt: skip
_MONO_KEYS = {DA_V2_LARGE: "depth_anything_v2_large", MOGE_3_VITL: "moge_3_vitl"}


def _loaders(device: str) -> dict[str, Callable[[], Any]]:
    def raft_net(name: str) -> Callable[[], Any]:
        def load() -> Any:
            from drone3d.keyframes.flow import load_raft

            return load_raft(name, device)

        return load

    def mono_net() -> Any:
        from drone3d.fastsfm.mono import load_mono

        return load_mono(DA_V2_LARGE, device)

    def moge_net() -> Any:
        from drone3d.fastsfm.mono import load_moge

        return load_moge(MOGE_3_VITL, device)

    def marigold_net() -> Any:
        from drone3d.depth.marigold import MarigoldDepth

        return MarigoldDepth(quantization="4bit", device=device)

    return {"raft_large": raft_net("raft_large"), "raft_small": raft_net("raft_small"),
            "depth_anything_v2_large": mono_net, "moge_3_vitl": moge_net, "marigold_v2": marigold_net}  # fmt: skip


@dataclass
class _Entry:
    status: str = "unloaded"  # unloaded | loading | loaded | error
    obj: Any = None
    vram_bytes: int = 0
    load_seconds: float | None = None
    loaded_at: float | None = None
    last_used: float | None = None
    uses: int = 0
    error: str | None = None


class ModelCache:
    """Loaded networks by key, their wrappers (with CUDA graphs) and bookkeeping."""

    def __init__(self, device: str = "cuda", reserve_gb: float = 2.0, *, concurrent: bool = False) -> None:
        self.device, self.reserve_gb = device, reserve_gb
        # With several engine slots, RAFT runs eagerly: capturing a CUDA graph while another slot launches
        # kernels fails ("unjoined work", "illegal state") and a failed capture left state behind that a later
        # kernel tripped over (a device-side assert on a two-slot engine). Graphs gain 1-27 % on one slot.
        self.cuda_graphs = not concurrent
        self.concurrent = concurrent
        self._lock = threading.RLock()
        self._entries = {k: _Entry() for k in SPECS}
        self._wrappers: dict[tuple, Any] = {}
        self._loaders = _loaders(device)
        self.busy = 0  # runs in progress; unloading waits for zero
        self._loading: dict[str, threading.Event] = {}
        self._pending_unload: set[str] = set()

    # ------------------------------------------------------------- lifecycle
    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for key, spec in SPECS.items():
                e = self._entries[key]
                out.append({"key": key, "title": spec.title, "role": spec.role, "need_gb": spec.need_gb,
                            "stages": list(spec.stages), "source": spec.source, "license": spec.license,
                            "status": e.status, "vram_mb": round(e.vram_bytes / 2**20), "load_seconds": e.load_seconds,
                            "loaded_at": e.loaded_at, "last_used": e.last_used, "uses": e.uses, "error": e.error,
                            "unload_pending": key in self._pending_unload})  # fmt: skip
            return out

    def load(self, key: str) -> _Entry:
        """Load ``key`` if needed; raises :class:`InsufficientMemory` when the GPU lacks room.

        The network is built outside the cache lock (seconds for Depth Anything,
        a minute for Marigold) so status requests never wait on it; a second
        request for the same model waits for the first load instead of starting one.
        """
        if key not in SPECS:
            raise KeyError(f"unknown model {key!r} ({', '.join(SPECS)})")
        while True:
            with self._lock:
                e = self._entries[key]
                self._pending_unload.discard(key)
                if e.status == "loaded":
                    return e
                pending = self._loading.get(key)
                if pending is None:
                    free = free_bytes()
                    need = (SPECS[key].need_gb + self.reserve_gb) * GiB
                    if free is not None and free < need:
                        e.status, e.error = (
                            "unloaded",
                            f"needs {need / GiB:.1f} GiB free, {free / GiB:.1f} GiB is",
                        )
                        raise InsufficientMemory(f"{SPECS[key].title}: {e.error}")
                    e.status, e.error = "loading", None
                    done = self._loading[key] = threading.Event()
                    break
            pending.wait()  # someone else is loading it; then look again
        import torch

        before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        t0 = time.perf_counter()
        try:
            obj = self._loaders[key]()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            with self._lock:
                e.status, e.obj, e.error = "error", None, f"{type(exc).__name__}: {exc}"
                self._loading.pop(key, None)
            done.set()
            raise
        with self._lock:
            e.obj = obj
            e.load_seconds = round(time.perf_counter() - t0, 2)
            e.vram_bytes = (
                (torch.cuda.memory_allocated() - before) if torch.cuda.is_available() else 0
            )
            e.status, e.loaded_at = "loaded", time.time()
            self._loading.pop(key, None)
        done.set()
        log.info("model %s loaded in %.2fs (%.2f GiB)", key, e.load_seconds, e.vram_bytes / GiB)
        return e

    def unload(self, key: str) -> str:
        """``"unloaded"``, ``"pending"`` (freed when the running job ends) or ``"not-loaded"``."""
        with self._lock:
            e = self._entries[key]
            if e.status != "loaded":
                return "not-loaded"
            if self.busy:
                self._pending_unload.add(key)
                return "pending"
            self._drop(key)
            return "unloaded"

    def unload_all(self) -> None:
        with self._lock:
            for key in SPECS:
                if self._entries[key].status == "loaded":
                    self._drop(key)

    def _drop(self, key: str) -> None:
        e = self._entries[key]
        obj, e.obj = e.obj, None
        self._wrappers = {k: v for k, v in self._wrappers.items() if k[0] != key}
        if key == "marigold_v2" and obj is not None:
            obj.close()
        del obj
        e.status, e.vram_bytes, e.loaded_at = "unloaded", 0, None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("model %s unloaded", key)

    @contextlib.contextmanager
    def job(self) -> Iterator[None]:
        """Mark a run in progress: unloads requested meanwhile wait until it ends."""
        with self._lock:
            self.busy += 1
        try:
            yield
        finally:
            with self._lock:
                self.busy -= 1
                if not self.busy:
                    for key in sorted(self._pending_unload):
                        if self._entries[key].status == "loaded":
                            self._drop(key)
                    self._pending_unload.clear()

    def _use(self, key: str) -> Any:
        e = self.load(key)  # outside the lock: see load()
        with self._lock:
            e.uses += 1
            e.last_used = time.time()
            return e.obj

    # ------------------------------------------------------------- accessors
    def raft(self, model: str, *, batch: int, iters: int) -> Any:
        from drone3d.keyframes.flow import RaftFlow

        net = self._use(model)
        with self._lock:
            k = (model, batch, iters, threading.get_ident())
            if (
                k not in self._wrappers
            ):  # one per batch size and engine slot: its CUDA graphs and buffers are its own
                self._wrappers[k] = RaftFlow(
                    model, batch=batch, iters=iters, device=self.device, net=net, cuda_graph=self.cuda_graphs
                )
            return self._wrappers[k]

    def mono(self, model: str, *, long_side: int, batch: int) -> Any:
        from drone3d.fastsfm.mono import mono_depth

        key = _MONO_KEYS.get(model)
        if key is None:  # another checkpoint: load it uncached rather than evict the warm one
            return mono_depth(model, device=self.device, long_side=long_side, batch=batch)
        return mono_depth(model, device=self.device, long_side=long_side, batch=batch, net=self._use(key))

    def marigold(self) -> Any:
        return self._use("marigold_v2")


_ACTIVE: ModelCache | None = None


_GPU = threading.RLock()


@contextlib.contextmanager
def gpu_exclusive() -> Iterator[None]:
    """On a multi-slot engine, one slot at a time runs GPU work; elsewhere a no-op.

    Two slots on the GPU at once kept meeting state shared inside libraries -- RAFT's
    correlation pyramid, nvJPEG's global coder, CUDA-graph capture, batched cuSOLVER --
    and a device-side assert still followed after each was fixed. Serialising the GPU
    phases keeps what a second slot is for: one run's CPU phases (mapping, writing
    files, decoding) overlap the other's GPU phases.
    """
    cache = _ACTIVE
    if cache is None or not getattr(cache, "concurrent", False):
        yield
        return
    with _GPU:
        yield


def activate(cache: ModelCache | None) -> None:
    """Make stages in this process draw from ``cache`` (None: build and free per use)."""
    global _ACTIVE
    _ACTIVE = cache


def active() -> ModelCache | None:
    return _ACTIVE


def raft(model: str = "raft_large", *, batch: int = 32, iters: int = 12) -> Any:
    """A :class:`~drone3d.keyframes.flow.RaftFlow`, warm when the engine holds one."""
    if _ACTIVE is None:
        from drone3d.keyframes.flow import RaftFlow

        return RaftFlow(model, batch=batch, iters=iters)
    return _ACTIVE.raft(model, batch=batch, iters=iters)


def mono(model: str = DA_V2_LARGE, *, long_side: int = 700, batch: int = 16) -> Any:
    """A :class:`~drone3d.fastsfm.mono.MonoDepth`, warm when the engine holds one."""
    if _ACTIVE is None:
        from drone3d.fastsfm.mono import mono_depth

        return mono_depth(model, long_side=long_side, batch=batch)
    return _ACTIVE.mono(model, long_side=long_side, batch=batch)


@contextlib.contextmanager
def marigold(*, quantization: str = "4bit", device: str = "cuda") -> Iterator[Any]:
    """Marigold v2; built and closed around the block unless the engine keeps a warm one."""
    if _ACTIVE is not None and quantization == "4bit":
        yield _ACTIVE.marigold()
        return
    from drone3d.depth.marigold import MarigoldDepth

    net = MarigoldDepth(quantization=quantization, device=device)
    try:
        yield net
    finally:
        net.close()
