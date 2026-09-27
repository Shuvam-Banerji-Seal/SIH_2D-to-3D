"""Importing drone3d undoes OpenMP binding that would pin the whole pipeline to one core."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity") or (os.cpu_count() or 1) < 2, reason="needs Linux, >1 CPU")
def test_import_restores_affinity_after_openmp_bound_the_process() -> None:
    code = (
        "import os, torch\n"
        "torch.ones(512, 512).sum()\n"  # starts OpenMP with OMP_PROC_BIND=spread: pins this thread
        "bound = len(os.sched_getaffinity(0))\n"
        "import drone3d\n"
        "torch.ones(512, 512).mm(torch.ones(512, 512))\n"  # the runtime must not re-pin it
        "print(bound, len(os.sched_getaffinity(0)), os.environ.get('OMP_PROC_BIND'))\n"
    )
    env = {**os.environ, "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300, check=True)
    bound, after, var = out.stdout.split()
    allowed = len(os.sched_getaffinity(0))
    assert int(bound) < allowed  # the problem is real in this environment
    assert int(after) == allowed and var == "None"
