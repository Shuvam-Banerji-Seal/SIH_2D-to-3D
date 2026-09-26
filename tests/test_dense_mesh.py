"""Tests for dense MVS and Open3D/Trimesh meshing backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.config import DenseConfig, MeshConfig
from drone3d.dense.mvs import ColmapMvsBackend
from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.mesh.texturing import (
    Open3DMesher,
    convert_mesh_format,
    has_open3d,
    has_trimesh,
    poisson_mesh_from_cloud,
)
from drone3d.types import SfMResult


def _sfm(tmp_path: Path) -> SfMResult:
    model = tmp_path / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(b"\x00")
    return SfMResult(backend="colmap", model_path=model)


# --- dense MVS -------------------------------------------------------------


def test_mvs_issues_the_three_colmap_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argvs: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> object:
        argv = [str(c) for c in cmd]
        argvs.append(argv)
        # stereo_fusion must find its output
        if argv[1] == "stereo_fusion":
            Path(argv[argv.index("--output_path") + 1]).write_bytes(b"ply")
        return None

    monkeypatch.setattr("drone3d.dense.mvs.run_command", fake_run)
    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.dense.mvs.ply_vertex_count", lambda _: 42)

    result = ColmapMvsBackend(binary="colmap").reconstruct(
        tmp_path / "images", _sfm(tmp_path), tmp_path / "out", DenseConfig()
    )

    assert [a[1] for a in argvs] == [
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion",
    ]
    assert result.num_points == 42
    assert result.fused_ply is not None


def test_mvs_passes_geom_consistency_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    argvs: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> object:
        argv = [str(c) for c in cmd]
        argvs.append(argv)
        if argv[1] == "stereo_fusion":
            Path(argv[argv.index("--output_path") + 1]).write_bytes(b"ply")
        return None

    monkeypatch.setattr("drone3d.dense.mvs.run_command", fake_run)
    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.dense.mvs.ply_vertex_count", lambda _: 1)

    ColmapMvsBackend().reconstruct(
        tmp_path / "images",
        _sfm(tmp_path),
        tmp_path / "out",
        DenseConfig(geom_consistency=False),
    )

    patch = next(a for a in argvs if a[1] == "patch_match_stereo")
    assert patch[patch.index("--PatchMatchStereo.geom_consistency") + 1] == "false"
    fusion = next(a for a in argvs if a[1] == "stereo_fusion")
    assert fusion[fusion.index("--input_type") + 1] == "photometric"


def test_mvs_requires_a_sparse_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: "/usr/bin/colmap")

    with pytest.raises(ReconstructionError, match="sparse model"):
        ColmapMvsBackend().reconstruct(
            tmp_path / "images", SfMResult(backend="x"), tmp_path / "out", DenseConfig()
        )


def test_mvs_missing_binary_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: None)

    with pytest.raises(BackendUnavailable):
        ColmapMvsBackend().reconstruct(
            tmp_path / "images", _sfm(tmp_path), tmp_path / "out", DenseConfig()
        )


def test_mvs_missing_fused_output_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("drone3d.dense.mvs.run_command", lambda *a, **k: None)
    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: "/usr/bin/colmap")

    with pytest.raises(ReconstructionError, match="fused.ply"):
        ColmapMvsBackend().reconstruct(
            tmp_path / "images", _sfm(tmp_path), tmp_path / "out", DenseConfig()
        )


def test_mvs_is_available_follows_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: "/usr/bin/colmap")
    assert ColmapMvsBackend().is_available() is True

    monkeypatch.setattr("drone3d.dense.mvs.which", lambda _: None)
    assert ColmapMvsBackend().is_available() is False


# --- meshing helpers -------------------------------------------------------


def test_has_open3d_and_trimesh_report_boolean() -> None:
    assert isinstance(has_open3d(), bool)
    assert isinstance(has_trimesh(), bool)


def test_poisson_mesh_rejects_empty_cloud(tmp_path: Path) -> None:
    if not has_open3d():
        pytest.skip("open3d not installed")
    cloud = tmp_path / "empty.ply"
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")

    with pytest.raises(ReconstructionError, match="empty"):
        poisson_mesh_from_cloud(cloud, tmp_path / "mesh.ply")


def test_convert_mesh_format_round_trip(tmp_path: Path) -> None:
    if not has_trimesh():
        pytest.skip("trimesh not installed")
    import trimesh

    source = tmp_path / "in.ply"
    trimesh.creation.icosphere(subdivisions=1).export(str(source))

    out = convert_mesh_format(source, tmp_path / "out.glb")

    assert out.is_file()
    assert out.stat().st_size > 0


def test_open3d_mesher_reports_availability() -> None:
    mesher = Open3DMesher()
    assert mesher.name == "open3d"
    assert mesher.is_available() == has_open3d()


def test_open3d_mesher_unavailable_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("drone3d.mesh.texturing.has_open3d", lambda: False)

    with pytest.raises(BackendUnavailable):
        Open3DMesher().build(
            tmp_path / "cloud.ply", tmp_path / "images", tmp_path / "out", MeshConfig()
        )
