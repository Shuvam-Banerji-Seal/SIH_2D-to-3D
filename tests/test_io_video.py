"""Regression tests for frame extraction (F4: budget on unknown-length video)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from drone3d.io.video import compute_frame_step, extract_frames
from drone3d.types import VideoInfo

FRAME_COUNT = 10
FPS = 10.0


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    """A real, decodable MJPG clip with a known frame count."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 24))
    assert writer.isOpened(), "cv2.VideoWriter failed to open MJPG/avi"
    for i in range(FRAME_COUNT):
        frame = np.full((24, 32, 3), i * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _fake_probe(video: Path, frame_count: int) -> VideoInfo:
    return VideoInfo(
        path=Path(video),
        width=32,
        height=24,
        fps=FPS,
        frame_count=frame_count,
        duration_s=frame_count / FPS if frame_count > 0 else 0.0,
        codec="MJPG",
    )


def test_compute_frame_step() -> None:
    assert compute_frame_step(FPS, 0.0) == 1  # sample_fps<=0 keeps every frame
    assert compute_frame_step(FPS, 5.0) == 2
    assert compute_frame_step(FPS, 2.0) == 5
    assert compute_frame_step(0.0, 2.0) == 1


# --- F4: max_frames must hold when frame_count is unknown (<= 0) ------------


def test_unknown_length_video_still_honours_max_frames(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path, tmp_path: Path
) -> None:
    """Before the fix, frame_count<=0 set `wanted=None` and wrote EVERY frame."""
    monkeypatch.setattr(
        "drone3d.io.video.probe_video", lambda p: _fake_probe(tiny_video, frame_count=0)
    )

    records = extract_frames(
        tiny_video,
        tmp_path / "out",
        sample_fps=0.0,
        max_frames=3,
        show_progress=False,
    )

    assert len(records) == 3


def test_unknown_length_video_still_honours_sample_fps(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path, tmp_path: Path
) -> None:
    """Before the fix, sample_fps was ignored outright when frame_count<=0."""
    monkeypatch.setattr(
        "drone3d.io.video.probe_video", lambda p: _fake_probe(tiny_video, frame_count=0)
    )

    records = extract_frames(
        tiny_video,
        tmp_path / "out",
        sample_fps=FPS / 2,  # step == 2 -> every other frame
        max_frames=600,
        show_progress=False,
    )

    assert [r.index for r in records] == [0, 2, 4, 6, 8]


# --- known-length path must be unchanged -----------------------------------


def test_known_length_video_decimates_to_budget(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "drone3d.io.video.probe_video",
        lambda p: _fake_probe(tiny_video, frame_count=FRAME_COUNT),
    )

    records = extract_frames(
        tiny_video,
        tmp_path / "out",
        sample_fps=0.0,
        max_frames=3,
        show_progress=False,
    )

    assert len(records) == 3
    # uniform decimation keeps the outermost candidates
    assert records[0].index == 0
    assert records[-1].index == FRAME_COUNT - 1


def test_known_length_video_samples_at_requested_rate(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "drone3d.io.video.probe_video",
        lambda p: _fake_probe(tiny_video, frame_count=FRAME_COUNT),
    )

    records = extract_frames(
        tiny_video,
        tmp_path / "out",
        sample_fps=FPS / 2,
        max_frames=600,
        show_progress=False,
    )

    assert [r.index for r in records] == [0, 2, 4, 6, 8]


def test_extracted_frames_are_written_to_disk(
    monkeypatch: pytest.MonkeyPatch, tiny_video: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "drone3d.io.video.probe_video",
        lambda p: _fake_probe(tiny_video, frame_count=FRAME_COUNT),
    )
    out_dir = tmp_path / "out"

    records = extract_frames(tiny_video, out_dir, sample_fps=0.0, max_frames=2, show_progress=False)

    assert len(records) == 2
    for record in records:
        assert record.path.is_file()
        assert record.path.parent == out_dir
        assert record.path.name == f"frame_{record.index:06d}.jpg"
