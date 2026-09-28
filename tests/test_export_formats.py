"""Exports round-trip: LAS coordinates and colour, GeoTIFF georeferencing, DSM rasterisation, meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drone3d.export.formats import NODATA, rasterize_top, write_geotiff, write_las, write_mesh


def test_rasterize_top_keeps_the_highest_point_per_cell() -> None:
    pts = np.array([[0.2, 0.2, 1.0], [0.3, 0.1, 5.0], [1.5, 0.5, 2.0], [0.5, 1.9, 3.0]])
    rgb = np.array([[10, 10, 10], [200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)
    dsm, ortho, (x0, y1) = rasterize_top(pts, rgb, cell=1.0)
    assert (x0, y1) == (0.2, 1.9)
    assert dsm.shape == (2, 2)
    # row 0 is the north edge (y from 1.9 down to 0.9): only the point at y = 1.9
    assert dsm[0, 0] == 3.0 and dsm[0, 1] == NODATA
    assert dsm[1, 0] == 5.0 and tuple(ortho[1, 0]) == (200, 0, 0)  # 5 m beats 1 m in the same cell
    assert dsm[1, 1] == 2.0


def test_las_round_trip(tmp_path: Path) -> None:
    import laspy

    rng = np.random.default_rng(1)
    pts = rng.uniform(-50, 50, (1000, 3)) + np.array([300000.0, 3000000.0, 200.0])
    rgb = rng.integers(0, 256, (1000, 3), dtype=np.uint8)
    path = write_las(pts, rgb, tmp_path / "p.las", epsg=32643)
    las = laspy.read(str(path))
    np.testing.assert_allclose(np.c_[las.x, las.y, las.z], pts, atol=6e-4)
    np.testing.assert_array_equal(np.asarray(las.red) // 257, rgb[:, 0])
    assert las.header.parse_crs().to_epsg() == 32643


def test_geotiff_carries_scale_tiepoint_and_crs(tmp_path: Path) -> None:
    import tifffile

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = write_geotiff(
        arr, tmp_path / "dsm.tif", origin=(500000.0, 3000100.0), cell=0.5, epsg=32643
    )
    with tifffile.TiffFile(path) as tif:
        np.testing.assert_array_equal(tif.asarray(), arr)
        geo = tif.geotiff_metadata
        assert geo["ModelPixelScale"][:2] == [0.5, 0.5]
        assert geo["ModelTiepoint"][3:5] == [500000.0, 3000100.0]
        assert int(geo["ProjectedCSTypeGeoKey"]) == 32643


def test_write_mesh_formats(tmp_path: Path) -> None:
    import trimesh

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    c = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], dtype=np.uint8)
    paths = write_mesh(v, f, c, tmp_path / "mesh")
    assert [p.suffix for p in paths] == [".ply", ".obj", ".glb"]
    back = trimesh.load(paths[2], force="mesh")
    assert len(back.faces) == 4
    np.testing.assert_array_equal(back.visual.vertex_colors[:, :3].min(0) <= 9, True)


def test_glb_is_y_up_and_stl_keeps_the_survey_frame(tmp_path: Path) -> None:
    import trimesh

    from drone3d.export.formats import Y_UP

    v = np.array([[1.0, 2.0, 30.0], [2.0, 2.0, 30.0], [1.0, 3.0, 31.0]])  # z = height
    f = np.array([[0, 1, 2]])
    glb, stl = write_mesh(v, f, None, tmp_path / "m", ("glb", "stl"))
    got = trimesh.load(glb, force="mesh").vertices
    np.testing.assert_allclose(sorted(got[:, 1]), sorted(v[:, 2]))  # height is glTF's +Y
    np.testing.assert_allclose(
        np.sort(got @ Y_UP, axis=0), np.sort(v, axis=0)
    )  # Y_UP is a rotation: undone by its transpose
    np.testing.assert_allclose(
        np.sort(trimesh.load(stl, force="mesh").vertices, axis=0), np.sort(v, axis=0), atol=1e-6
    )
    assert abs(np.linalg.det(Y_UP) - 1) < 1e-12


def test_blend_is_skipped_without_blender(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from drone3d.export import formats

    monkeypatch.setattr(formats, "blender_binary", lambda: None)
    assert formats.write_blend(tmp_path / "m.glb", tmp_path / "m.blend") is None


def test_frame_previews_for_the_viewer(tmp_path: Path) -> None:
    from PIL import Image

    from drone3d.export.stage import _frame_previews

    class Im:
        def __init__(self, name: str) -> None:
            self.name = name

    images, depth = tmp_path / "images", tmp_path / "depth"
    (images / "pass_00").mkdir(parents=True)
    depth.mkdir()
    posed = [Im(f"pass_00/f_{k:06d}.jpg") for k in range(0, 100, 5)]
    for im in posed:
        Image.new("RGB", (1920, 1080), (100, 120, 140)).save(images / im.name)
    Image.new("RGB", (160, 90)).save(depth / "f_000020.jpg")
    out = _frame_previews(posed, images, depth, tmp_path / "frames", most=6)
    assert len(out["names"]) == 5 and out["photo"] and out["depth"]  # every 4th of 20
    with Image.open(tmp_path / "frames" / "photo" / "f_000000.jpg") as im:
        assert max(im.size) == 320
    assert (tmp_path / "frames" / "depth" / "f_000020.jpg").is_file()
    assert _frame_previews([], images, depth, tmp_path / "x") is None


def test_decimation_keeps_shape_and_colour(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    import open3d as o3d

    from drone3d.export.stage import _decimate

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=80)
    v, f = np.asarray(sphere.vertices), np.asarray(sphere.triangles)
    vc = np.where(v[:, 2:3] > 0, [[200, 40, 40]], [[40, 40, 200]]).astype(
        np.uint8
    )  # red north, blue south
    sphere.vertex_colors = o3d.utility.Vector3dVector(vc / 255.0)
    for fast in (True, False):
        if not fast:  # the Open3D fallback, as without fast-simplification installed
            monkeypatch.setitem(sys.modules, "fast_simplification", None)
        dv, df, dvc = _decimate(sphere, v, f, vc, 2000)
        assert 1800 <= len(df) <= 2000
        np.testing.assert_allclose(np.linalg.norm(dv, axis=1), 2.0, atol=0.08)  # still the sphere
        assert (dvc[dv[:, 2] > 0.5, 0] > 150).all() and (dvc[dv[:, 2] < -0.5, 2] > 150).all()


def test_decimation_keeps_colours_where_they_were() -> None:
    import pytest

    pytest.importorskip("fast_simplification")
    from drone3d.export.stage import _decimate

    n = 80  # a gently curved sheet, red growing along x
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([xs.ravel(), ys.ravel(), 0.05 * np.sin(3 * xs.ravel())], 1)
    q = np.arange(n * n).reshape(n, n)
    f = np.concatenate([np.stack([q[:-1, :-1].ravel(), q[1:, :-1].ravel(), q[:-1, 1:].ravel()], 1),
                        np.stack([q[1:, :-1].ravel(), q[1:, 1:].ravel(), q[:-1, 1:].ravel()], 1)])  # fmt: skip
    vc = np.stack([np.round(255 * v[:, 0]), np.full(len(v), 90), np.full(len(v), 160)], 1).astype(np.uint8)
    dv, df, dc = _decimate(None, v, f, vc, len(f) // 4)
    assert abs(len(df) - len(f) // 4) <= 2 and dc.shape == (len(dv), 3)
    assert np.abs(dc[:, 0].astype(float) - 255 * dv[:, 0]).max() < 12  # the colour of the place it moved to
