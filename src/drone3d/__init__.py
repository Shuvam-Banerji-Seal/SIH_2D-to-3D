"""drone3d: single-pass drone video to georeferenced, metric 3D reconstruction.

Built for Smart India Hackathon problem statement 26158 (NTRO):
"Single-Pass Drone Video to Accurate 3D Model Generation System".
"""

import os as _os


def _unbind_threads() -> None:
    """Undo OpenMP thread binding inherited from the shell.

    With ``OMP_PROC_BIND`` set (``spread``/``close``/``true``), the first OpenMP
    runtime to start in the process -- torch, pycolmap, Open3D -- pins the main
    thread to one core, and every thread and subprocess started afterwards
    (Ceres, the spirula trainer and mesher, ffmpeg) inherits that single core.
    On a shared host this ran the whole pipeline's CPU work on core 0: the
    global mapper got 0.13 cores until re-pinned, then 5.3. The variables are
    dropped for this process and its children only, and the affinity is reset
    in case a runtime already bound this thread.
    """
    for var in ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
        _os.environ.pop(var, None)
    if hasattr(_os, "sched_setaffinity"):
        import contextlib

        with contextlib.suppress(OSError):  # a cgroup cpuset may forbid some cores: keep ours
            _os.sched_setaffinity(0, range(_os.cpu_count() or 1))


_unbind_threads()

from drone3d.version import __version__  # noqa: E402

__all__ = ["__version__"]
