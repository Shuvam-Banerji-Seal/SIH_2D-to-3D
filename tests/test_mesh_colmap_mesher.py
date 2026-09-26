"""Regression tests for COLMAP meshing, including the Poisson crash fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.config import MeshConfig
from drone3d.exceptions import ReconstructionError
from drone3d.mesh.colmap_mesher import ColmapMesher


@pytest.fixture()
def dense_ply(tmp_path: Path) -> Path:
    path = tmp_path / "dense" / "fused.ply"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    return path


def test_poisson_falls_back_to_delaunay_on_crash(
    monkeypatch: pytest.MonkeyPatch, dense_ply: Path, tmp_path: Path
) -> None:
    """A crashing poisson_mesher must not kill the run (PS criterion 7)."""
    attempts: list[str] = []

    def fake_run_command(cmd: list[str], **_: object) -> object:
        argv = [str(item) for item in cmd]
        attempts.append(argv[1])
        if argv[1] == "poisson_mesher":
            raise ReconstructionError("command failed (exit -11): colmap poisson_mesher")
        Path(argv[argv.index("--output_path") + 1]).write_bytes(b"mesh")
        return None

    monkeypatch.setattr("drone3d.mesh.colmap_mesher.run_command", fake_run_command)
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.ply_element_counts", lambda _: (3, 1))

    result = ColmapMesher(method="poisson", binary="colmap").build(
        dense_ply, tmp_path / "images", tmp_path / "out", MeshConfig(texture=False)
    )

    assert attempts == ["poisson_mesher", "delaunay_mesher"]
    assert result.backend == "colmap-delaunay"
    assert result.mesh_path is not None
    assert result.mesh_path.name == "mesh-delaunay.ply"


def test_delaunay_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, dense_ply: Path, tmp_path: Path
) -> None:
    """Only Poisson gets a fallback; other failures must surface."""

    def fake_run_command(cmd: list[str], **_: object) -> object:
        raise ReconstructionError("command failed (exit 1)")

    monkeypatch.setattr("drone3d.mesh.colmap_mesher.run_command", fake_run_command)
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")

    with pytest.raises(ReconstructionError):
        ColmapMesher(method="delaunay", binary="colmap").build(
            dense_ply, tmp_path / "images", tmp_path / "out", MeshConfig(texture=False)
        )


def test_delaunay_command_has_no_output_type_flag(
    monkeypatch: pytest.MonkeyPatch, dense_ply: Path, tmp_path: Path
) -> None:
    """F15: COLMAP 4.x rejects `--output_type` on delaunay_mesher."""
    argvs: list[list[str]] = []

    def fake_run_command(cmd: list[str], **_: object) -> object:
        argv = [str(item) for item in cmd]
        argvs.append(argv)
        Path(argv[argv.index("--output_path") + 1]).write_bytes(b"mesh")
        return None

    monkeypatch.setattr("drone3d.mesh.colmap_mesher.run_command", fake_run_command)
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.ply_element_counts", lambda _: (3, 1))

    ColmapMesher(method="delaunay", binary="colmap").build(
        dense_ply, tmp_path / "images", tmp_path / "out", MeshConfig(texture=False)
    )

    assert "--output_type" not in argvs[0]
    assert argvs[0][1] == "delaunay_mesher"
    assert argvs[0][-1].endswith("mesh-delaunay.ply")


def test_missing_dense_cloud_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")

    with pytest.raises(ReconstructionError):
        ColmapMesher(method="delaunay", binary="colmap").build(
            tmp_path / "nope.ply",
            tmp_path / "images",
            tmp_path / "out",
            MeshConfig(texture=False),
        )
