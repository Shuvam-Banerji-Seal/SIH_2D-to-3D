"""Tests for the command-line entry point."""

from __future__ import annotations

import http.server
import json
import sys
from pathlib import Path

import pytest

from drone3d import cli
from drone3d.cli import main


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0

    out = capsys.readouterr().out
    assert "drone3d" in out or out.strip()


def test_top_level_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip()


def test_doctor_runs_and_reports(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "spirula" in out
    assert "ffmpeg" in out
    assert "optional extras" in out


def test_doctor_exits_nonzero_and_names_a_missing_core_dependency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    present = cli._has_module
    monkeypatch.setattr(cli, "_has_module", lambda name: name != "tifffile" and present(name))

    assert main(["doctor"]) == 1

    captured = capsys.readouterr()
    assert "MISSING tifffile" in captured.out
    assert "tifffile" in captured.err


def test_doctor_runs_without_torch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # It once imported the torch-importing Marigold module just to read a path, and crashed on
    # an install without the gpu extra -- the very install it is meant to diagnose.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "drone3d.depth.marigold", raising=False)

    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "torch not installed" in out
    assert "marigold" in out


def test_viewer_serves_export_formats_without_touching_the_stdlib_table() -> None:
    handler_class = cli._viewer_handler()
    handler = handler_class.__new__(handler_class)  # guess_type needs no connection

    assert handler.guess_type("export/model.glb") == "model/gltf-binary"
    assert handler.guess_type("export/index.html") == "text/html"  # what the stdlib knows is kept
    assert handler.guess_type("export/model.ply.gz") == "application/gzip"
    assert ".glb" not in http.server.SimpleHTTPRequestHandler.extensions_map


def test_view_without_an_export_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["view", str(tmp_path)]) == 1
    assert "no viewer" in capsys.readouterr().err


def test_init_config_writes_yaml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "my.yaml"

    assert main(["init-config", str(target)]) == 0

    text = target.read_text(encoding="utf-8")
    assert "ingest:" in text
    assert "sfm:" in text
    assert "wrote default configuration" in capsys.readouterr().out


def test_init_config_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "existing.yaml"
    target.write_text("original: content\n")

    assert main(["init-config", str(target)]) == 1
    assert target.read_text(encoding="utf-8") == "original: content\n"


def test_init_config_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "existing.yaml"
    target.write_text("original: content\n")

    assert main(["init-config", str(target), "--force"]) == 0
    assert "original: content" not in target.read_text(encoding="utf-8")


def test_init_config_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "cfg.yaml"

    assert main(["init-config", str(target)]) == 0
    assert target.is_file()


def test_run_on_synthetic_clip_completes(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (64, 48))
    assert writer.isOpened()
    for i in range(6):
        frame = np.full((48, 64, 3), 60 + i * 20, dtype=np.uint8)
        frame[10:40, (i * 8) % 32 : (i * 8) % 32 + 24] = 230
        writer.write(frame)
    writer.release()

    config = tmp_path / "cfg.yaml"
    config.write_text(f"ingest:\n  video: {clip}\n")
    run_dir = tmp_path / "run"

    code = main(
        [
            "run",
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--stages",
            "ingest",
            "--quiet",
        ]
    )

    assert code == 0
    assert (run_dir / "manifest.json").is_file()
    info = json.loads((run_dir / "ingest" / "result.json").read_text())["video"]
    assert (info["width"], info["height"], info["num_frames"]) == (64, 48, 6)


def test_unknown_stage_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("ingest:\n  video: x.avi\n")

    code = main(
        ["run", "--config", str(config), "--run-dir", str(tmp_path / "r"), "--stages", "bogus"]
    )

    assert code != 0


def test_run_with_missing_video_fails_gracefully(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text(f"ingest:\n  video: {tmp_path / 'nope.avi'}\n")

    code = main(
        [
            "run",
            "--config",
            str(config),
            "--run-dir",
            str(tmp_path / "r"),
            "--stages",
            "ingest",
            "--quiet",
        ]
    )

    assert code != 0
