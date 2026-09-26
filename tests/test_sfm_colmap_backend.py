"""Regression tests for COLMAP command construction and model selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.config import SfMConfig
from drone3d.sfm.colmap_backend import ColmapSfMBackend, _largest_model


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the argv of every COLMAP invocation instead of running it."""
    calls: list[list[str]] = []

    def fake_run_command(cmd: list[str], **_: object) -> object:
        calls.append([str(item) for item in cmd])
        return None

    monkeypatch.setattr("drone3d.sfm.colmap_backend.run_command", fake_run_command)
    return calls


def _backend() -> ColmapSfMBackend:
    return ColmapSfMBackend(binary="colmap")


# --- F2: dynamic masks must reach COLMAP ------------------------------------


def test_extract_features_passes_mask_path_when_masks_exist(
    captured: list[list[str]], tmp_path: Path
) -> None:
    images_dir = tmp_path / "frames_selected"
    mask_dir = tmp_path / "preprocess" / "masks"
    images_dir.mkdir()
    mask_dir.mkdir(parents=True)
    # COLMAP convention: mask name == image name + ".png"
    (mask_dir / "frame_000000.jpg.png").write_bytes(b"\x00")

    _backend()._extract_features(
        "colmap", images_dir, tmp_path / "db.sqlite", SfMConfig(), mask_dir=mask_dir
    )

    argv = captured[0]
    assert "--ImageReader.mask_path" in argv
    assert argv[argv.index("--ImageReader.mask_path") + 1] == str(mask_dir)


def test_extract_features_omits_mask_path_without_masks(
    captured: list[list[str]], tmp_path: Path
) -> None:
    images_dir = tmp_path / "frames_selected"
    empty_masks = tmp_path / "preprocess" / "masks"
    images_dir.mkdir()
    empty_masks.mkdir(parents=True)

    _backend()._extract_features(
        "colmap", images_dir, tmp_path / "db.sqlite", SfMConfig(), mask_dir=empty_masks
    )
    _backend()._extract_features(
        "colmap", images_dir, tmp_path / "db.sqlite", SfMConfig(), mask_dir=None
    )

    assert "--ImageReader.mask_path" not in captured[0]
    assert "--ImageReader.mask_path" not in captured[1]


def test_extract_features_keeps_extra_args_last(captured: list[list[str]], tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    config = SfMConfig(extra_args=["--SiftExtraction.use_gpu", "0"])

    _backend()._extract_features(
        "colmap", images_dir, tmp_path / "db.sqlite", config, mask_dir=None
    )

    argv = captured[0]
    assert argv[-2:] == ["--SiftExtraction.use_gpu", "0"]


# --- F3: model selection contract -------------------------------------------


def test_largest_model_picks_directory_with_images_bin(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    (sparse / "0").mkdir(parents=True)
    (sparse / "1").mkdir()
    (sparse / "0" / "images.bin").write_bytes(b"\x00" * 10)
    (sparse / "1" / "images.bin").write_bytes(b"\x00" * 100)

    assert _largest_model(sparse) == sparse / "1"


def test_largest_model_accepts_text_only_model(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    (sparse / "0").mkdir(parents=True)
    (sparse / "0" / "images.txt").write_text("1\n2\n3\n")

    assert _largest_model(sparse) == sparse / "0"


def test_largest_model_rejects_plain_files_and_empty_dirs(tmp_path: Path) -> None:
    """A top-level `images.txt` FILE must not be mistaken for a model directory.

    F3 was a boolean-precedence hardening of this predicate. The written form
    `A and B or C` cannot currently be distinguished from `A and (B or C)` on a
    real filesystem (a non-directory cannot contain children), so this is a
    contract test for the intended semantics rather than a crash reproducer.
    """
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "images.txt").write_text("1\n")
    (sparse / "images.bin").write_bytes(b"\x00")
    (sparse / "0").mkdir()  # a directory with no model files

    assert _largest_model(sparse) is None


def test_largest_model_on_empty_dir_returns_none(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    sparse.mkdir()

    assert _largest_model(sparse) is None


# --- F13: TXT export needs its output directory to exist -------------------


def test_reconstruct_mkdirs_txt_output_dir_before_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`colmap model_converter --output_type TXT` aborts unless the dir exists.

    COLMAP writes cameras/images/points3D(.txt) *into* the output path and
    CHECK-fails with `std::invalid_argument` when it is missing. PLY output
    writes a single file and has no such requirement.
    """
    seen: list[tuple[str, bool]] = []

    def fake_run_command(cmd: list[str], **_: object) -> object:
        argv = [str(item) for item in cmd]
        if "--output_type" in argv:
            out_dir = Path(argv[argv.index("--output_path") + 1])
            seen.append((argv[argv.index("--output_type") + 1], out_dir.is_dir()))
        return None

    monkeypatch.setattr("drone3d.sfm.colmap_backend.run_command", fake_run_command)
    monkeypatch.setattr("drone3d.sfm.colmap_backend.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr(
        "drone3d.sfm.colmap_backend._largest_model", lambda _: tmp_path / "sparse" / "0"
    )
    monkeypatch.setattr(
        "drone3d.sfm.colmap_backend._read_model_stats",
        lambda _: {
            "num_images": 0,
            "num_points": 0,
            "mean_reprojection_error": None,
        },
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    _backend().reconstruct(images_dir, tmp_path / "sfm", SfMConfig(), mask_dir=None)

    txt = [existed for kind, existed in seen if kind == "TXT"]
    assert txt, "expected a TXT model conversion to be attempted"
    assert txt == [True], "sparse_txt must exist BEFORE model_converter runs"
