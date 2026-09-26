"""Regression tests for PLY I/O (F16: interleaved binary layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone3d.exceptions import IngestionError
from drone3d.utils.ply import load_ply, ply_vertex_count, read_ply_header, write_ply


def test_round_trip_without_colors(tmp_path: Path) -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.5, -6.25, 7.75], [0.0, 0.0, 0.0]])

    path = write_ply(tmp_path / "cloud.ply", points)
    cloud = load_ply(path)

    np.testing.assert_allclose(cloud.points, points, rtol=1e-6)
    assert cloud.colors is None


def test_round_trip_with_colors(tmp_path: Path) -> None:
    """F16: colours must be interleaved per vertex, not written as one block.

    Writing the colour block after all vertices made the reader (which expects
    a structured x,y,z,r,g,b record) read every vertex past the first
    misaligned -- silently corrupting the geometry.
    """
    points = np.array([[1.0, 2.0, 3.0], [4.5, -6.25, 7.75], [0.0, 0.0, 0.0], [-1.0, 100.0, 0.5]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [12, 34, 56]], dtype=np.uint8)

    path = write_ply(tmp_path / "cloud.ply", points, colors)
    cloud = load_ply(path)

    np.testing.assert_allclose(cloud.points, points, rtol=1e-6)
    assert cloud.colors is not None
    np.testing.assert_array_equal(cloud.colors, colors)


def test_round_trip_many_vertices_are_all_finite(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    points = rng.normal(scale=50.0, size=(500, 3))
    colors = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)

    path = write_ply(tmp_path / "cloud.ply", points, colors)
    cloud = load_ply(path)

    assert len(cloud.points) == 500
    assert np.isfinite(cloud.points).all()
    np.testing.assert_allclose(cloud.points, points, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(cloud.colors, colors)


def test_header_declares_interleaved_properties(tmp_path: Path) -> None:
    points = np.zeros((2, 3))
    colors = np.zeros((2, 3), dtype=np.uint8)

    path = write_ply(tmp_path / "cloud.ply", points, colors)
    header = read_ply_header(path)

    assert header["format"] == "binary_little_endian"
    names = [p["name"] for p in header["elements"][0]["properties"]]
    assert names == ["x", "y", "z", "red", "green", "blue"]


def test_vertex_count(tmp_path: Path) -> None:
    path = write_ply(tmp_path / "cloud.ply", np.zeros((7, 3)))
    assert ply_vertex_count(path) == 7


def test_load_rejects_missing_xyz(tmp_path: Path) -> None:
    bad = tmp_path / "bad.ply"
    bad.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float w\nend_header\n0\n")

    with pytest.raises(IngestionError):
        load_ply(bad)


def test_load_rejects_missing_vertex_element(tmp_path: Path) -> None:
    bad = tmp_path / "bad.ply"
    bad.write_text("ply\nformat ascii 1.0\nelement face 0\nend_header\n")

    with pytest.raises(IngestionError):
        load_ply(bad)


def test_color_length_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        write_ply(tmp_path / "cloud.ply", np.zeros((3, 3)), np.zeros((2, 3), np.uint8))
