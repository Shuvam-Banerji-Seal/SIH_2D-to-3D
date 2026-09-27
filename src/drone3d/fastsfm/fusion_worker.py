"""TSDF fusion and mesh extraction in a child process with a CUDA context of its own.

Open3D's GPU mesh extraction can fail with an illegal memory access or an
"unspecified launch failure" after earlier grids in the same process -- a
sticky CUDA error that poisons every later CUDA call there. In ``drone3d run``
a process lives for one video; in the warm engine it lives for many, and a
46k-block grid (well inside the limits of :func:`~drone3d.fastsfm.dense.tsdf_fuse`)
crashed the engine after a dozen videos. Fusion therefore runs in a worker
process, started when the dense stage starts (so it boots while the first depth
maps are computed): a fault kills only the worker, the model is retried once
in a fresh one, and the caller's CUDA context -- and its warm networks -- are
never touched by Open3D's allocator.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from typing import Any

import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["FusionError", "FusionWorker"]

log = get_logger(__name__)


class FusionError(RuntimeError):
    """Fusion failed twice (in two fresh workers) for one model."""


def _serve(conn) -> None:  # type: ignore[no-untyped-def]
    import drone3d  # noqa: F401  (unpins OpenMP threads, see drone3d/__init__.py)
    from drone3d.fastsfm.dense import tsdf_fuse

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        frames, kw = msg
        try:
            import open3d.core as o3c

            vbg, voxel = tsdf_fuse(frames, **kw)
            active, capacity = int(vbg.hashmap().size()), int(vbg.hashmap().capacity())
            mesh = vbg.extract_triangle_mesh().to_legacy()
            pcd = vbg.extract_point_cloud().to_legacy()
            out = {
                "voxel": float(voxel), "active": active, "capacity": capacity,
                "vertices": np.asarray(mesh.vertices, dtype=np.float64),
                "triangles": np.asarray(mesh.triangles, dtype=np.int32),
                "vertex_colors": np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None,
                "points": np.asarray(pcd.points, dtype=np.float64),
                "point_colors": np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None,
            }  # fmt: skip
            del vbg, mesh, pcd
            if o3c.cuda.is_available():
                o3c.cuda.release_cache()
            conn.send(out)
        except Exception as exc:  # report, then exit: after a CUDA fault this process is unusable
            conn.send({"error": f"{type(exc).__name__}: {str(exc)[:400]}"})
            return


class FusionWorker:
    """``fuse(frames, **tsdf_kwargs)`` -> arrays of the mesh and point cloud, computed in a child process."""

    def __init__(self, *, start: bool = True) -> None:
        self._ctx = mp.get_context("spawn")  # a fresh interpreter: no CUDA state inherited
        self._proc = None
        self._conn = None
        self.restarts = 0
        if start:
            self._start()

    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_serve, args=(child,), name="drone3d-fusion", daemon=True
        )
        self._proc.start()
        child.close()
        self._conn = parent

    def _stop(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(OSError, BrokenPipeError):
                self._conn.send(None)
            self._conn.close()
        if self._proc is not None:
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.kill()
        self._proc = self._conn = None

    def fuse(self, frames: list, **kw: Any) -> dict:
        errors = []
        for attempt in range(2):
            if self._proc is None or not self._proc.is_alive():
                if self._proc is not None:
                    self.restarts += 1
                self._stop()
                self._start()
            try:
                self._conn.send((frames, kw))
                out = self._conn.recv()
            except (EOFError, OSError, BrokenPipeError) as exc:  # the worker died (segfault, abort)
                out = {"error": f"fusion worker died: {type(exc).__name__}"}
            if "error" not in out:
                return out
            errors.append(out["error"])
            log.warning(
                "TSDF fusion failed (attempt %d): %s; retrying in a fresh process",
                attempt + 1,
                out["error"][:160],
            )
            self._stop()
        raise FusionError("; ".join(errors))

    def close(self) -> None:
        self._stop()

    def __enter__(self) -> FusionWorker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
