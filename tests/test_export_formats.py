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
    path = write_geotiff(arr, tmp_path / "dsm.tif", origin=(500000.0, 3000100.0), cell=0.5, epsg=32643)
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
