"""Tests for the subprocess wrapper every external backend goes through."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from drone3d.exceptions import ReconstructionError
from drone3d.utils.shell import run_command, which


def test_run_command_success() -> None:
    result = run_command([sys.executable, "-c", "print('hello')"])

    assert result.returncode == 0
    assert "hello" in result.stdout


def test_run_command_raises_on_nonzero_exit() -> None:
    with pytest.raises(ReconstructionError) as excinfo:
        run_command([sys.executable, "-c", "import sys; sys.exit(3)"])

    message = str(excinfo.value)
    assert "command failed" in message
    assert "exit 3" in message


def test_run_command_error_carries_stderr_tail() -> None:
    script = "import sys; sys.stderr.write('boom-detail\\n'); sys.exit(2)"

    with pytest.raises(ReconstructionError, match="boom-detail"):
        run_command([sys.executable, "-c", script])


def test_run_command_check_false_returns_result() -> None:
    result = run_command([sys.executable, "-c", "import sys; sys.exit(4)"], check=False)

    assert result.returncode == 4


def test_run_command_missing_binary_raises() -> None:
    with pytest.raises(ReconstructionError, match="executable not found"):
        run_command(["/nonexistent/definitely-not-a-binary", "--version"])


def test_run_command_timeout_raises() -> None:
    with pytest.raises(ReconstructionError, match="timed out"):
        run_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.3)


def test_run_command_accepts_path_args(tmp_path: Path) -> None:
    marker = tmp_path / "sub"
    marker.mkdir()

    result = run_command([sys.executable, "-c", "import os; print(os.getcwd())"], cwd=marker)

    assert Path(result.stdout.strip()).resolve() == marker.resolve()


def test_run_command_accepts_pathlib_parts(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('from-path')\n")

    result = run_command([sys.executable, script])

    assert "from-path" in result.stdout


def test_which_resolves_and_misses() -> None:
    assert which(sys.executable) is not None
    assert which("definitely-not-installed-xyz") is None
