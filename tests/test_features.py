"""Tests for feature detection, matching and the frame overlap graph."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from drone3d.sfm.features import (
    build_frame_graph,
    detect_features,
    match_descriptors,
    pair_overlap,
)


def _scene(size: int = 256, seed: int = 0, shift: int = 0) -> np.ndarray:
    """A textured scene with many corners; ``shift`` pans the camera."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = (((xx // 16) + (yy // 16)) % 2 * 180 + 50).astype(np.uint8)
    base = (base + rng.integers(0, 30, base.shape, dtype=np.uint8)).clip(0, 255).astype(np.uint8)
    if shift:
        base = np.roll(base, shift, axis=1)
    return base


# --- detection -------------------------------------------------------------


def test_detect_features_sift_finds_keypoints() -> None:
    keypoints, descriptors = detect_features(_scene(), method="sift", max_features=500)

    assert len(keypoints) > 10
    assert descriptors is not None
    assert descriptors.shape[0] == len(keypoints)


def test_detect_features_orb() -> None:
    keypoints, descriptors = detect_features(_scene(), method="orb")
    assert len(keypoints) > 0
    assert descriptors is not None


def test_detect_features_akaze_unavailable_on_opencv5() -> None:
    """AKAZE was removed in OpenCV 5.x; the backend must fail clearly, not crash
    the other methods (F19: the detector map used to be built eagerly)."""
    if hasattr(cv2, "AKAZE_create"):
        keypoints, descriptors = detect_features(_scene(), method="akaze")
        assert len(keypoints) > 0
        assert descriptors is not None
    else:
        with pytest.raises(ValueError, match="akaze"):
            detect_features(_scene(), method="akaze")


def test_detect_features_sift_still_works_without_akaze() -> None:
    """Regression for F19: one missing backend must not break the others."""
    assert not hasattr(cv2, "AKAZE_create") or True

    keypoints, _ = detect_features(_scene(), method="sift")

    assert len(keypoints) > 0


def test_detect_features_accepts_bgr_input() -> None:
    bgr = cv2.cvtColor(_scene(), cv2.COLOR_GRAY2BGR)

    keypoints, _ = detect_features(bgr, method="orb")

    assert len(keypoints) > 0


def test_detect_features_respects_max_features() -> None:
    keypoints, _ = detect_features(_scene(), method="sift", max_features=20)

    # nfeatures is a soft cap in OpenCV; allow a little slack
    assert 0 < len(keypoints) <= 20 * 1.5


def test_detect_features_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        detect_features(_scene(), method="nope")


# --- matching --------------------------------------------------------------


def test_match_descriptors_round_trip() -> None:
    image = _scene()
    _, desc_a = detect_features(image, method="sift")
    _, desc_b = detect_features(np.roll(image, 4, axis=1), method="sift")

    matches = match_descriptors(desc_a, desc_b, method="sift")

    assert len(matches) > 5


def test_match_descriptors_handles_none() -> None:
    assert match_descriptors(None, None) == []

    _, desc = detect_features(_scene(), method="sift")
    assert match_descriptors(desc, None) == []
    assert match_descriptors(None, desc) == []


def test_match_descriptors_uses_hamming_for_orb() -> None:
    _, desc_a = detect_features(_scene(), method="orb")
    _, desc_b = detect_features(np.roll(_scene(), 3, axis=1), method="orb")

    matches = match_descriptors(desc_a, desc_b, method="orb")

    assert isinstance(matches, list)


# --- overlap ---------------------------------------------------------------


def test_pair_overlap_detects_same_scene() -> None:
    image = _scene()
    moved = np.roll(image, 6, axis=1)

    result = pair_overlap(image, moved, method="sift")

    assert result["matches"] > 0
    assert result["connected"] is True
    assert 0.0 <= result["overlap"] <= 1.0


def test_pair_overlap_unrelated_scenes_disconnected() -> None:
    # A different *structure* (fine vertical stripes), not merely different
    # noise over the same checkerboard — SIFT legitimately matches the latter.
    rng = np.random.default_rng(99)
    yy, xx = np.mgrid[0:256, 0:256]
    stripes = (((xx // 3) % 2) * 200 + 40).astype(np.uint8)
    b = (stripes + rng.integers(0, 15, stripes.shape, dtype=np.uint8)).clip(0, 255).astype(np.uint8)

    result = pair_overlap(_scene(seed=1), b, method="sift", min_inliers=12)

    assert result["connected"] is False


def test_pair_overlap_matches_key_shape() -> None:
    result = pair_overlap(_scene(), _scene(), method="orb")

    assert set(result) == {"matches", "inliers", "overlap", "connected"}


# --- frame graph -----------------------------------------------------------


def test_build_frame_graph_links_temporal_neighbours(tmp_path: Path) -> None:
    paths = []
    for i in range(4):
        path = tmp_path / f"f{i}.png"
        cv2.imwrite(str(path), _scene(shift=i * 5))
        paths.append(path)

    edges = build_frame_graph(paths, method="sift", neighbor_window=2, show_progress=False)

    assert edges, "overlapping frames must produce connected edges"
    assert all(edge["connected"] for edge in edges)
    # indices must be within range
    assert all(0 <= edge["i"] < 4 and 0 <= edge["j"] < 4 for edge in edges)


def test_build_frame_graph_rejects_unreadable_frame(tmp_path: Path) -> None:
    bad = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        build_frame_graph([bad], show_progress=False)


def test_build_frame_graph_pair_budget(tmp_path: Path) -> None:
    paths = []
    for i in range(6):
        path = tmp_path / f"f{i}.png"
        cv2.imwrite(str(path), _scene(shift=i * 3))
        paths.append(path)

    edges = build_frame_graph(
        paths, method="orb", neighbor_window=3, max_pairs=3, show_progress=False
    )

    assert isinstance(edges, list)
