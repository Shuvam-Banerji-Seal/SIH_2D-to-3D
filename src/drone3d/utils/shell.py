"""Thin, logged wrapper around :mod:`subprocess` for external binaries (COLMAP, ffmpeg)."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from drone3d.exceptions import ReconstructionError
from drone3d.logging_utils import get_logger

__all__ = ["run_command", "which"]

log = get_logger(__name__)


def which(executable: str) -> str | None:
    """Return the resolved path of ``executable`` or ``None``."""
    import shutil

    return shutil.which(executable)


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
    timeout_s: float | None = None,
    check: bool = True,
    log_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing output and raising on failure.

    Raises:
        ReconstructionError: If the process exits non-zero and ``check`` is set,
            or the binary is missing.
    """
    args = [str(part) for part in command]
    log.debug("exec: %s", shlex.join(args))
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ReconstructionError(f"executable not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ReconstructionError(f"command timed out after {timeout_s}s: {args[0]}") from exc

    if log_output and result.stderr:
        for line in result.stderr.strip().splitlines()[-5:]:
            log.debug("%s: %s", args[0], line)
    if check and result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise ReconstructionError(
            f"command failed (exit {result.returncode}): {shlex.join(args)}\n{stderr_tail}"
        )
    return result
