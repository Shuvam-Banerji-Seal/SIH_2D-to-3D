"""Tests for dense MVS and Open3D/Trimesh meshing backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.config import DenseConfig, MeshConfig
from drone3d.dense.base import DenseBackend, get_dense_backend
from drone3d.dense.mvs import ColmapMvsBackend
from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.mesh.base import MeshBackend, get_mesh_backend
from drone3d.mesh.colmap_mesher import ColmapMesher
from drone3d.mesh.texturing import (
    Open3DMesher,
    convert_mesh_format,
    has_open3d,
    has_trimesh,
    poisson_mesh_from_cloud,
)
from drone3d.sfm.base import SfMBackend, get_sfm_backend
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
            Path(argv[argv.index("--output_path") + 1]).write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
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
            Path(argv[argv.index("--output_path") + 1]).write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
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


def test_texture_step_uses_mesh_texturer_not_the_old_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F25: COLMAP 4.x renamed `texture_mesher` to `mesh_texturer`.

    The old name exits "command not recognized" while the stage reported ok and
    produced no textured mesh -- and a textured mesh is criterion 4.
    """
    argvs: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> object:
        argv = [str(c) for c in cmd]
        argvs.append(argv)
        if argv[1] == "mesh_texturer":
            out = Path(argv[argv.index("--output_path") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "mesh.ply").write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
        elif argv[1] == "delaunay_mesher":
            Path(argv[argv.index("--output_path") + 1]).write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
        return None

    monkeypatch.setattr("drone3d.mesh.colmap_mesher.run_command", fake_run)
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.ply_element_counts", lambda _: (3, 1))

    dense = tmp_path / "dense" / "fused.ply"
    dense.parent.mkdir(parents=True)
    dense.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )

    result = ColmapMesher(method="delaunay", binary="colmap").build(
        dense, tmp_path / "images", tmp_path / "out", MeshConfig(texture=True)
    )

    commands = [a[1] for a in argvs]
    assert "mesh_texturer" in commands
    assert "texture_mesher" not in commands
    assert result.textured_mesh_path is not None
    assert result.metadata["textured"] is True


def test_texture_step_uses_the_dense_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`mesh_texturer` needs the undistorter workspace, not the raw images dir."""
    argvs: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> object:
        argv = [str(c) for c in cmd]
        argvs.append(argv)
        if argv[1] == "mesh_texturer":
            out = Path(argv[argv.index("--output_path") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "mesh.ply").write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
        elif argv[1] == "delaunay_mesher":
            Path(argv[argv.index("--output_path") + 1]).write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
        return None

    monkeypatch.setattr("drone3d.mesh.colmap_mesher.run_command", fake_run)
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.ply_element_counts", lambda _: (3, 1))

    dense = tmp_path / "dense" / "fused.ply"
    dense.parent.mkdir(parents=True)
    dense.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )

    ColmapMesher(method="delaunay").build(
        dense, tmp_path / "images", tmp_path / "out", MeshConfig(texture=True)
    )

    tex = next(a for a in argvs if a[1] == "mesh_texturer")
    assert Path(tex[tex.index("--workspace_path") + 1]) == dense.parent


def test_every_dense_factory_returns_a_dense_backend_subclass() -> None:
    """F26: ColmapMvsBackend/MonoDepthBackend declared the ABC's interface but
    never inherited from it, so `isinstance(x, DenseBackend)` was False while
    the factory was typed `-> DenseBackend`."""
    for name in ("mvs", "mono", "none"):
        backend = get_dense_backend(name)
        assert isinstance(backend, DenseBackend), name


def test_all_factories_return_true_subclasses() -> None:
    """F26: the factories are typed `-> XBackend` but the dense backends did not
    actually inherit `DenseBackend`, so `isinstance` checks silently failed and
    the declared return type was a lie. Every factory must return a real
    subclass of its ABC."""
    from drone3d.dense.base import DenseBackend

    for name in ("mvs", "mono", "none"):
        assert isinstance(get_dense_backend(name), DenseBackend), name
    for name in ("poisson", "delaunay", "open3d", "none"):
        assert isinstance(get_mesh_backend(name), MeshBackend), name
    assert isinstance(get_sfm_backend("colmap"), SfMBackend)


def test_mesher_rejects_empty_dense_cloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F27: an empty fused.ply passes is_file but crashes COLMAP's meshers.

    The `accurate` profile produced a 0-point fused.ply; poisson_mesher then
    SIGSEGV'd and the delaunay fallback SIGABRT'd. The mesher must reject the
    empty cloud up front with a clear error.
    """
    empty = tmp_path / "fused.ply"
    empty.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")

    with pytest.raises(ReconstructionError, match="empty"):
        ColmapMesher(method="delaunay").build(
            empty, tmp_path / "images", tmp_path / "out", MeshConfig(texture=False)
        )


def test_default_mesher_emits_obj_for_ps_deliverable3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F31: PS deliverable 3 names `mesh.obj`, but ColmapMesher only wrote PLY.

    `convert_mesh_format` could always produce OBJ; it was just never called on
    the default path (only Open3DMesher used it, and it wrote .glb)."""
    argvs: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> object:
        argv = [str(c) for c in cmd]
        argvs.append(argv)
        if argv[1] == "mesh_texturer":
            out = Path(argv[argv.index("--output_path") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "mesh.ply").write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                b"property float x\nproperty float y\nproperty float z\n"
                b"end_header\n" + b"\x00" * 12
            )
        elif argv[1] == "delaunay_mesher":
            Path(argv[argv.index("--output_path") + 1]).write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                b"property float x\nproperty float y\nproperty float z\n"
                b"end_header\n" + b"\x00" * 12
            )
        return None

    monkeypatch.setattr("drone3d.mesh.colmap_mesher.run_command", fake_run)
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.which", lambda _: "/usr/bin/colmap")
    monkeypatch.setattr("drone3d.mesh.colmap_mesher.ply_element_counts", lambda _: (1, 0))

    dense = tmp_path / "dense" / "fused.ply"
    dense.parent.mkdir(parents=True)
    dense.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n" + b"\x00" * 12
    )

    result = ColmapMesher(method="delaunay").build(
        dense, tmp_path / "images", tmp_path / "out", MeshConfig(texture=True)
    )

    if "obj_path" in result.metadata and result.metadata["obj_path"]:
        assert Path(result.metadata["obj_path"]).is_file()
        assert result.metadata["obj_path"].endswith("mesh.obj")
    else:
        # trimesh absent -- the field must at least be present and None
        assert "obj_path" in result.metadata
