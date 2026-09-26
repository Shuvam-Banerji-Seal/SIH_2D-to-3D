"""Tests for COLMAP TXT model parsing (feeds the georeferencing stage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone3d.exceptions import ReconstructionError
from drone3d.sfm.colmap_model import (
    ImagePose,
    parse_images_text,
    parse_points3d,
    qvec_to_rotation,
)

IMAGES_TXT = """\
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_000000.jpg
128.5 256.25 1

2 0.70710678 0.0 0.70710678 0.0 -1.0 -2.0 -3.0 1 frame_000024.jpg
10.0 20.0 -1
"""

POINTS3D_TXT = """\
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 1.5 -2.5 3.5 255 0 0 0.25 1 0
2 10.0 20.0 30.0 0 255 0 0.5 2 1
"""


@pytest.fixture()
def images_txt(tmp_path: Path) -> Path:
    path = tmp_path / "images.txt"
    path.write_text(IMAGES_TXT)
    return path


@pytest.fixture()
def points3d_txt(tmp_path: Path) -> Path:
    path = tmp_path / "points3D.txt"
    path.write_text(POINTS3D_TXT)
    return path


# --- quaternion / pose -----------------------------------------------------


def test_qvec_identity_is_identity_matrix() -> None:
    np.testing.assert_allclose(
        qvec_to_rotation(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3), atol=1e-12
    )


def test_qvec_is_normalised() -> None:
    rotation = qvec_to_rotation(np.array([2.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)


def test_qvec_90deg_about_z() -> None:
    half = np.sqrt(0.5)
    rotation = qvec_to_rotation(np.array([half, 0.0, 0.0, half]))

    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_qvec_rotation_is_orthonormal() -> None:
    rotation = qvec_to_rotation(np.array([0.5, 0.5, 0.5, 0.5]))

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_degenerate_quaternion_raises() -> None:
    with pytest.raises(ReconstructionError, match="quaternion"):
        qvec_to_rotation(np.zeros(4))


def test_pose_rotation_property() -> None:
    pose = ImagePose(
        image_id=1,
        name="a.jpg",
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.array([1.0, 2.0, 3.0]),
        camera_id=1,
    )

    np.testing.assert_allclose(pose.rotation, np.eye(3), atol=1e-12)


def test_pose_center_is_minus_R_transpose_t() -> None:
    """COLMAP stores world-to-camera: centre = -R^T @ t."""
    qvec = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])  # 90 deg about z
    tvec = np.array([1.0, 0.0, 0.0])
    pose = ImagePose(image_id=1, name="a.jpg", qvec=qvec, tvec=tvec, camera_id=1)

    expected = -qvec_to_rotation(qvec).T @ tvec

    np.testing.assert_allclose(pose.center, expected, atol=1e-12)


# --- images.txt ------------------------------------------------------------


def test_parse_images_text_reads_poses_only(images_txt: Path) -> None:
    poses = parse_images_text(images_txt)

    assert len(poses) == 2
    assert poses[0].name == "frame_000000.jpg"
    assert poses[0].image_id == 1
    assert poses[0].camera_id == 1
    assert poses[1].name == "frame_000024.jpg"
    np.testing.assert_allclose(poses[1].tvec, [-1.0, -2.0, -3.0])


def test_parse_images_text_keeps_names_with_spaces(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text("1 1 0 0 0 0 0 0 1 my frame name.jpg\n")

    poses = parse_images_text(path)

    assert poses[0].name == "my frame name.jpg"


def test_parse_images_text_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text("# comment\n\n1 1 0 0 0 0 0 0 1 a.jpg\n")

    assert len(parse_images_text(path)) == 1


def test_parse_images_text_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text("1 1 0 0 0 0 0 0 1 ok.jpg\nnot numbers at all\n2 1 0 0 0 0 0 0 1 also_ok.jpg\n")

    poses = parse_images_text(path)

    assert [p.name for p in poses] == ["ok.jpg", "also_ok.jpg"]


def test_parse_images_text_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionError):
        parse_images_text(tmp_path / "nope.txt")


# --- points3D.txt ----------------------------------------------------------


def test_parse_points3d(points3d_txt: Path) -> None:
    xyz, rgb = parse_points3d(points3d_txt)

    assert xyz.shape == (2, 3)
    assert rgb.shape == (2, 3)
    np.testing.assert_allclose(xyz[0], [1.5, -2.5, 3.5])
    np.testing.assert_array_equal(rgb[1], [0, 255, 0])


def test_parse_points3d_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionError):
        parse_points3d(tmp_path / "nope.txt")
