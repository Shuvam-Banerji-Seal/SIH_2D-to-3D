"""Exports in the formats problem statement 26158 lists: OBJ, PLY, LAS, GeoTIFF, glTF/GLB, FBX.

Everything takes points or a mesh in one Cartesian frame: local ENU metres
after georeferencing, or the SfM frame without GPS. With an EPSG code (a UTM
zone) the LAS and GeoTIFF files carry it; without one they are written in
the local frame and say so.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["rasterize_top", "write_fbx", "write_geotiff", "write_las", "write_mesh"]

log = get_logger(__name__)
NODATA = -9999.0


def write_mesh(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None, stem: Path,
               formats: tuple[str, ...] = ("ply", "obj", "glb")) -> list[Path]:  # fmt: skip
    """Mesh with optional per-vertex colour (uint8 or [0, 1] float) -> ``stem.<fmt>`` files."""
    import trimesh

    vc = None
    if colors is not None:
        c = np.asarray(colors)
        vc = (np.clip(c, 0, 1) * 255).astype(np.uint8) if c.dtype.kind == "f" else c.astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), vertex_colors=vc, process=False)
    out = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        mesh.export(path)
        out.append(path)
    return out


def write_las(points: np.ndarray, rgb: np.ndarray | None, path: Path, *, epsg: int | None = None,
              scale: float = 0.001) -> Path:  # fmt: skip
    """LAS 1.4, point format 7 (RGB), millimetre quantisation; CRS as WKT when ``epsg`` is given."""
    import laspy

    header = laspy.LasHeader(point_format=7, version="1.4")
    pts = np.asarray(points, dtype=np.float64)
    header.offsets = np.floor(pts.min(0))
    header.scales = np.array([scale, scale, scale])
    if epsg is not None:
        import pyproj

        header.add_crs(pyproj.CRS.from_epsg(epsg))
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    if rgb is not None:
        c = np.asarray(rgb)
        c16 = (np.clip(c, 0, 1) * 65535).astype(np.uint16) if c.dtype.kind == "f" else c.astype(np.uint16) * 257
        las.red, las.green, las.blue = c16[:, 0], c16[:, 1], c16[:, 2]
    las.write(str(path))
    return path


def rasterize_top(points: np.ndarray, rgb: np.ndarray | None, cell: float) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float]]:
    """Top-down raster of the highest point per ``cell`` -> ``(dsm [H, W], ortho [H, W, 3] | None, (x_min, y_max))``.

    Row 0 is the northern (max y) edge, as GeoTIFF expects; empty cells are ``NODATA``.
    """
    p = np.asarray(points, dtype=np.float64)
    x0, y1 = p[:, 0].min(), p[:, 1].max()
    col = np.floor((p[:, 0] - x0) / cell).astype(np.int64)
    row = np.floor((y1 - p[:, 1]) / cell).astype(np.int64)
    w, h = int(col.max()) + 1, int(row.max()) + 1
    flat = row * w + col
    order = np.lexsort((p[:, 2], flat))  # by cell, then height: the last per cell is the top
    last = np.r_[flat[order][1:] != flat[order][:-1], True]
    top = order[last]
    dsm = np.full(h * w, NODATA, dtype=np.float32)
    dsm[flat[top]] = p[top, 2]
    ortho = None
    if rgb is not None:
        c = np.asarray(rgb)
        c8 = (np.clip(c, 0, 1) * 255).astype(np.uint8) if c.dtype.kind == "f" else c.astype(np.uint8)
        ortho = np.zeros((h * w, 3), dtype=np.uint8)
        ortho[flat[top]] = c8[top]
        ortho = ortho.reshape(h, w, 3)
    return dsm.reshape(h, w), ortho, (float(x0), float(y1))


def write_geotiff(array: np.ndarray, path: Path, *, origin: tuple[float, float], cell: float,
                  epsg: int | None = None, nodata: float | None = NODATA) -> Path:  # fmt: skip
    """North-up GeoTIFF with pixel scale and tie point; projected EPSG CRS or a user-defined local one."""
    import tifffile

    x0, y1 = origin
    # GeoKeyDirectory: version 1.1.0; GTModelType (1024), GTRasterType (1025, PixelIsArea),
    # ProjectedCSType (3072) = EPSG or 32767 (user-defined), ProjLinearUnits (3076) = metre.
    keys = [1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg or 32767, 3076, 0, 1, 9001]
    tags = [
        (33550, "d", 3, (cell, cell, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, x0, y1, 0.0), False),
        (34735, "H", len(keys), keys, False),
    ]
    if nodata is not None:
        tags.append((42113, "s", 0, f"{nodata:g}", False))
    photometric = "rgb" if array.ndim == 3 else "minisblack"
    tifffile.imwrite(path, array, photometric=photometric, extratags=tags, compression="zlib")
    return path


def write_fbx(src: Path, dst: Path, converter: str | None = None) -> Path | None:
    """Convert a mesh file to binary FBX with assimp (``tools/build_meshconv.sh``), if built."""
    root = Path(__file__).resolve().parents[3]
    exe = converter or next((str(p) for p in (root / ".tools" / "bin" / "meshconv",) if p.is_file()), None)
    if exe is None:
        log.warning("FBX skipped: build .tools/bin/meshconv with tools/build_meshconv.sh")
        return None
    res = subprocess.run([exe, str(src), str(dst), "fbx"], capture_output=True, text=True, check=False)
    if res.returncode != 0 or not dst.is_file():
        log.warning("FBX export failed: %s", (res.stderr or res.stdout)[-300:])
        return None
    return dst
