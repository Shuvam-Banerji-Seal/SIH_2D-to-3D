"""End-to-end smoke test on a synthetic clip (PS criterion 7: no crash).

No external binaries are required: COLMAP is disabled via ``sfm.backend=none``
so the downstream stages must *degrade gracefully* rather than abort, which is
exactly the robustness behaviour the problem statement scores.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone3d.config import load_config
from drone3d.pipeline import Pipeline

FRAMES = 12
FPS = 6.0
SIZE = (96, 72)  # width, height


def _synthetic_clip(path: Path) -> Path:
    """A small MJPG clip with enough variation to pass quality filtering."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    assert writer.isOpened()
    rng = np.random.default_rng(4)
    for i in range(FRAMES):
        frame = np.full((SIZE[1], SIZE[0], 3), 40 + i * 8, dtype=np.uint8)
        # moving high-frequency block keeps Laplacian variance well above the
        # blur threshold on every frame
        x = (i * 7) % 48
        frame[24:60, x : x + 40] = 220
        frame += rng.integers(0, 6, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_ingest_and_preprocess_on_synthetic_clip(tmp_path: Path) -> None:
    clip = _synthetic_clip(tmp_path / "clip.avi")
    config = load_config(
        None,
        [f"ingest.video={clip}", "preprocess.max_frames=5", "preprocess.sample_fps=0"],
    )

    result = Pipeline(config, run_dir=tmp_path / "run").run(["ingest", "preprocess", "report"])

    statuses = {report.name: report.status for report in result.stages}
    assert statuses["ingest"] == "ok"
    assert statuses["preprocess"] == "ok"
    assert statuses["report"] == "ok"

    assert (tmp_path / "run" / "ingest" / "frames.csv").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_preprocess_respects_frame_budget(tmp_path: Path) -> None:
    clip = _synthetic_clip(tmp_path / "clip.avi")
    config = load_config(
        None,
        [f"ingest.video={clip}", "preprocess.max_frames=3", "preprocess.sample_fps=0"],
    )

    result = Pipeline(config, run_dir=tmp_path / "run").run(["ingest", "preprocess"])

    report = next(r for r in result.stages if r.name == "preprocess")
    assert report.status == "ok"
    assert report.metrics["selected"] <= 3


def test_missing_backend_degrades_gracefully(tmp_path: Path) -> None:
    """Criterion 7: an unavailable backend skips, it must not crash the run."""
    clip = _synthetic_clip(tmp_path / "clip.avi")
    config = load_config(
        None,
        [
            f"ingest.video={clip}",
            "preprocess.sample_fps=0",
            "sfm.backend=none",
            "sfm.allow_missing=true",
        ],
    )

    result = Pipeline(config, run_dir=tmp_path / "run").run()

    statuses = {report.name: report.status for report in result.stages}
    # ingest/preprocess must still succeed...
    assert statuses["ingest"] == "ok"
    assert statuses["preprocess"] == "ok"
    # ...while SfM-dependent stages skip instead of failing
    assert statuses["sfm"] in {"skipped", "ok"}
    for name in ("dense", "mesh", "georef"):
        assert statuses[name] in {"skipped", "ok"}, f"{name} should not crash"

    # the run as a whole is still considered successful
    assert result.ok is True
    assert (tmp_path / "run" / "report.html").is_file()


def test_stage_subset_runs_independently(tmp_path: Path) -> None:
    """Any stage can be re-run on its own against an existing run directory."""
    clip = _synthetic_clip(tmp_path / "clip.avi")
    config = load_config(None, [f"ingest.video={clip}", "preprocess.sample_fps=0"])
    run_dir = tmp_path / "run"

    Pipeline(config, run_dir=run_dir).run(["ingest"])
    second = Pipeline(config, run_dir=run_dir).run(["preprocess"])

    assert all(report.status in {"ok", "skipped"} for report in second.stages)
    assert (run_dir / "frames_selected").is_dir()
