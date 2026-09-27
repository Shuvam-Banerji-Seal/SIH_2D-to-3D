"""A trainer that stops writing is killed, not waited on forever."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from drone3d.splat.spirula import STALLED, _run


def test_a_silent_process_is_killed_after_the_stall_limit(tmp_path: Path) -> None:
    t0 = time.monotonic()
    code, text, _ = _run([sys.executable, "-c", "print('step 1', flush=True); import time; time.sleep(60)"],
                         tmp_path / "t.log", stall_s=1.5)  # fmt: skip
    assert code == STALLED and time.monotonic() - t0 < 20
    assert "step 1" in text and "killed" in text


def test_a_process_that_keeps_writing_is_left_alone(tmp_path: Path) -> None:
    prog = "import time\nfor i in range(4):\n    print(i, flush=True); time.sleep(0.8)"
    code, text, _ = _run([sys.executable, "-c", prog], tmp_path / "t.log", stall_s=1.5)
    assert code == 0 and text.strip().endswith("3")
