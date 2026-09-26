"""Tests for the HTML report content and the backend factories.

The report was previously only asserted to *exist*; these tests check that it
actually carries the run's numbers, which is what PS criterion 9 promises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.config import DenseConfig, MeshConfig, SfMConfig
from drone3d.dense.base import NullDenseBackend, get_dense_backend
from drone3d.exceptions import BackendUnavailable, ConfigError
from drone3d.mesh.base import NullMeshBackend, get_mesh_backend
from drone3d.report.html import generate_report, make_contact_sheet
from drone3d.sfm.base import NullSfMBackend, get_sfm_backend
from drone3d.types import Artifact, PipelineResult, StageReport

# --- factories -------------------------------------------------------------


def test_sfm_factory_names() -> None:
    assert isinstance(get_sfm_backend("none"), NullSfMBackend)
    assert isinstance(get_sfm_backend("null"), NullSfMBackend)
    assert get_sfm_backend("colmap").name == "colmap"


def test_sfm_factory_rejects_unknown() -> None:
    with pytest.raises(ConfigError, match="unknown SfM backend"):
        get_sfm_backend("bogus")


def test_null_sfm_backend_raises_on_use() -> None:
    with pytest.raises(BackendUnavailable, match="set to 'none'"):
        NullSfMBackend().reconstruct(Path("x"), Path("y"), SfMConfig())


def test_dense_factory_names() -> None:
    from drone3d.dense.mvs import ColmapMvsBackend

    assert isinstance(get_dense_backend("none"), NullDenseBackend)
    assert get_dense_backend("mvs").name == "mvs"
    assert get_dense_backend("mono").name == "mono"
    # binary kwarg is threaded through to the MVS backend
    mvs = get_dense_backend("mvs", binary="/opt/colmap")
    assert isinstance(mvs, ColmapMvsBackend)
    assert mvs.binary == "/opt/colmap"


def test_dense_factory_rejects_unknown() -> None:
    with pytest.raises(ConfigError, match="unknown dense backend"):
        get_dense_backend("bogus")


def test_null_dense_backend_raises_on_use() -> None:
    from drone3d.types import SfMResult

    with pytest.raises(BackendUnavailable, match="set to 'none'"):
        NullDenseBackend().reconstruct(Path("x"), SfMResult(backend="n"), Path("y"), DenseConfig())


def test_mesh_factory_names() -> None:
    from drone3d.mesh.colmap_mesher import ColmapMesher

    assert isinstance(get_mesh_backend("none"), NullMeshBackend)
    poisson = get_mesh_backend("poisson")
    delaunay = get_mesh_backend("delaunay")
    assert isinstance(poisson, ColmapMesher) and poisson.method == "poisson"
    assert isinstance(delaunay, ColmapMesher) and delaunay.method == "delaunay"
    assert get_mesh_backend("open3d").name == "open3d"


def test_mesh_factory_rejects_unknown() -> None:
    with pytest.raises(ConfigError, match="unknown mesh backend"):
        get_mesh_backend("bogus")


def test_null_mesh_backend_raises_on_use() -> None:
    with pytest.raises(BackendUnavailable, match="set to 'none'"):
        NullMeshBackend().build(Path("x"), Path("y"), Path("z"), MeshConfig())


# --- report content (criterion 9 substance) --------------------------------


def _result() -> PipelineResult:
    result = PipelineResult(run_dir=Path())
    result.stages.append(StageReport("ingest", "ok", "20 frames sampled", duration_s=7.22))
    result.stages.append(StageReport("georef", "skipped", "no frame GPS fixes", duration_s=0.0))
    result.stages[-1].artifacts.append(Artifact("camera_track", Path("georef/track.geojson")))
    return result


def test_report_contains_stage_rows_and_messages(tmp_path: Path) -> None:
    path = generate_report(
        run_dir=tmp_path,
        result=_result(),
        context={"metrics": {"clouds": {"dense": {"n_points": 246354}}}},
    )

    html = path.read_text(encoding="utf-8")

    assert "ingest" in html
    assert "georef" in html
    assert "20 frames sampled" in html
    assert "no frame GPS fixes" in html
    # statuses are rendered (and colour-coded) rather than dropped
    assert ">ok<" in html or "ok</span>" in html
    assert ">skipped<" in html or "skipped</span>" in html


def test_report_carries_the_metrics_numbers(tmp_path: Path) -> None:
    path = generate_report(
        run_dir=tmp_path,
        result=_result(),
        context={"metrics": {"clouds": {"dense": {"n_points": 246354}}}},
    )

    html = path.read_text(encoding="utf-8")

    assert "246354" in html
    assert "n_points" in html
    assert "clouds" in html


def test_report_lists_artifacts(tmp_path: Path) -> None:
    path = generate_report(run_dir=tmp_path, result=_result(), context={})

    html = path.read_text(encoding="utf-8")

    assert "camera_track" in html
    assert "track.geojson" in html


def test_report_defaults_to_run_dir(tmp_path: Path) -> None:
    path = generate_report(run_dir=tmp_path, result=_result(), context={})

    assert path == tmp_path / "report.html"
    assert path.is_file()


def test_report_honours_out_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "custom.html"

    path = generate_report(run_dir=tmp_path, result=_result(), context={}, out_path=target)

    assert path == target
    assert target.is_file()


def test_report_escapes_untrusted_messages(tmp_path: Path) -> None:
    result = PipelineResult(run_dir=Path())
    result.stages.append(StageReport("ingest", "failed", "<script>alert(1)</script>"))

    html = generate_report(run_dir=tmp_path, result=result, context={}).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_is_standalone_html(tmp_path: Path) -> None:
    html = generate_report(run_dir=tmp_path, result=_result(), context={}).read_text(
        encoding="utf-8"
    )

    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in html


# --- contact sheet ---------------------------------------------------------


def test_make_contact_sheet_builds_an_image(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    paths: list[Path] = []
    for i in range(3):
        path = tmp_path / f"f{i}.png"
        cv2.imwrite(str(path), np.full((24, 32, 3), 40 * i + 20, dtype=np.uint8))
        paths.append(path)

    sheet = make_contact_sheet(paths, out_path=tmp_path / "sheet.png")

    assert sheet is not None
    assert sheet.is_file()
    assert sheet.stat().st_size > 0


def test_make_contact_sheet_handles_no_frames(tmp_path: Path) -> None:
    """An empty run must still produce a report rather than crash."""
    sheet = make_contact_sheet([], out_path=tmp_path / "empty.png")

    assert sheet is None or Path(sheet).is_file()
