"""Minimal, dependency-free PLY point-cloud reader/writer (ASCII + binary LE)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drone3d.exceptions import IngestionError

__all__ = ["PointCloud", "load_ply", "ply_vertex_count", "read_ply_header", "write_ply"]

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class PointCloud:
    """XYZ points with optional RGB colours in ``[0, 255]``."""

    points: np.ndarray
    colors: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.points.shape[0])


def _read_header(stream) -> dict:
    header = {"format": None, "elements": []}
    while True:
        raw = stream.readline()
        if not raw:
            raise IngestionError("PLY header ended unexpectedly")
        line = raw.decode("ascii", errors="replace").strip()
        if line == "end_header":
            break
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format":
            header["format"] = parts[1]
        elif parts[0] == "element":
            header["elements"].append({"name": parts[1], "count": int(parts[2]), "properties": []})
        elif parts[0] == "property" and header["elements"]:
            if parts[1] == "list":
                header["elements"][-1]["properties"].append(
                    {"list": True, "count_type": parts[2], "type": parts[3], "name": parts[4]}
                )
            else:
                header["elements"][-1]["properties"].append(
                    {"list": False, "type": parts[1], "name": parts[2]}
                )
    return header


def read_ply_header(path: str | Path) -> dict:
    """Parse and return a PLY header as a dictionary."""
    ply_path = Path(path)
    if not ply_path.is_file():
        raise IngestionError(f"PLY file not found: {ply_path}")
    with ply_path.open("rb") as stream:
        return _read_header(stream)


def load_ply(path: str | Path) -> PointCloud:
    """Load XYZ (and RGB if present) from a PLY file.

    Supports ``ascii`` and ``binary_little_endian`` formats. List properties
    (faces) are skipped.
    """
    ply_path = Path(path)
    if not ply_path.is_file():
        raise IngestionError(f"PLY file not found: {ply_path}")

    with ply_path.open("rb") as stream:
        header = _read_header(stream)
        if header["format"] not in {"ascii", "binary_little_endian"}:
            raise IngestionError(f"unsupported PLY format: {header['format']}")

        vertex = next((e for e in header["elements"] if e["name"] == "vertex"), None)
        if vertex is None:
            raise IngestionError(f"PLY has no vertex element: {ply_path}")
        scalar = [p for p in vertex["properties"] if not p["list"]]
        names = [p["name"] for p in scalar]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise IngestionError(f"PLY vertex element is missing '{axis}': {ply_path}")

        if header["format"] == "ascii":
            rows = []
            for _ in range(vertex["count"]):
                line = stream.readline()
                if not line:
                    break
                rows.append(line.split())
            data = np.array(rows, dtype=np.float64)
            index = {name: position for position, name in enumerate(names)}
            points = data[:, [index["x"], index["y"], index["z"]]]
            colors = _extract_ascii_colors(data, index)
        else:
            dtype = np.dtype([(p["name"], _PLY_TYPES[p["type"]]) for p in scalar])
            raw = np.frombuffer(stream.read(dtype.itemsize * vertex["count"]), dtype=dtype)
            points = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)
            color_names = [name for name in ("red", "green", "blue") if name in raw.dtype.names]
            colors = (
                np.column_stack([raw[name] for name in color_names]).astype(np.uint8)
                if len(color_names) == 3
                else None
            )

    return PointCloud(points=points, colors=colors)


def _extract_ascii_colors(data: np.ndarray, index: dict[str, int]) -> np.ndarray | None:
    if not {"red", "green", "blue"} <= set(index):
        return None
    return data[:, [index["red"], index["green"], index["blue"]]].astype(np.uint8)


def ply_vertex_count(path: str | Path) -> int:
    """Read only the header to return the vertex count."""
    header = read_ply_header(path)
    vertex = next((e for e in header["elements"] if e["name"] == "vertex"), None)
    return int(vertex["count"]) if vertex else 0


def write_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    """Write a binary little-endian PLY file."""
    ply_path = Path(path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3).astype(np.uint8)
        if len(colors) != len(points):
            raise IngestionError("colors and points must have the same length")

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment generated by drone3d",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    with ply_path.open("wb") as handle:
        handle.write(header_bytes)
        handle.write(points.astype("<f4").tobytes())
        if colors is not None:
            handle.write(colors.astype("u1").tobytes())
    return ply_path
