"""Gaussian splats for the web: a trained 3DGS PLY -> the compact ``.splat`` layout browsers load.

A 3DGS PLY stores 62 floats per splat (248 bytes: position, normal, SH degree
3, opacity, log-scales, rotation); the web format keeps 32 bytes -- position
and scale as float32, colour and opacity as RGBA8 (spherical-harmonic DC term
only), rotation as four bytes -- sorted by importance so a truncated file
still holds the splats that matter. The splats are also moved into the
model's export frame (levelled or georeferenced ENU), the frame of its mesh
and point cloud, so one viewer shows them together.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["compose", "export_splat", "read_3dgs_ply", "train_from_world", "write_splat"]

SH_C0 = 0.28209479177387814


def read_3dgs_ply(path: str | Path) -> dict[str, np.ndarray]:
    """``{"xyz", "f_dc", "opacity", "scale", "rot"}`` (raw trained values) from a binary 3DGS PLY."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: expected a binary little-endian PLY")
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    props = [line.split() for line in header if line.startswith("property")]
    if any(p[1] != "float" for p in props):
        raise ValueError(f"{path}: expected float properties only")
    names = [p[2] for p in props]
    data = np.frombuffer(raw, dtype="<f4", count=count * len(names), offset=end).reshape(
        count, len(names)
    )
    col = {n: i for i, n in enumerate(names)}

    def cols(*keys: str) -> np.ndarray:
        return data[:, [col[k] for k in keys]]

    return {
        "xyz": cols("x", "y", "z"),
        "f_dc": cols("f_dc_0", "f_dc_1", "f_dc_2"),
        "opacity": data[:, col["opacity"]],
        "scale": cols("scale_0", "scale_1", "scale_2"),
        "rot": cols("rot_0", "rot_1", "rot_2", "rot_3"),  # (w, x, y, z)
    }


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    m = np.asarray(r, dtype=np.float64)
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of (w, x, y, z) quaternions; ``a`` [4], ``b`` [N, 4]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=1)  # fmt: skip


def train_from_world(scene_transform: str | Path | None) -> tuple[float, np.ndarray, np.ndarray]:
    """``(s, R, t)`` with ``p_train = s R p_world + t`` from spirula's ``scene_transform.json`` (identity if absent)."""
    if scene_transform is None or not Path(scene_transform).is_file():
        return 1.0, np.eye(3), np.zeros(3)
    tf = json.loads(Path(scene_transform).read_text())["train_from_world"]
    return (
        float(tf["scale"]),
        np.asarray(tf["rotation"]["matrix_3x3"], float),
        np.asarray(tf["translation"], float),
    )


def compose(
    export: tuple[float, np.ndarray, np.ndarray], train: tuple[float, np.ndarray, np.ndarray]
):  # type: ignore[no-untyped-def]
    """Train frame -> export frame: ``export ∘ train_from_world⁻¹`` as one ``(s, R, t)``."""
    se, re_, te = export
    st, rt, tt = train
    s = se / st
    r = re_ @ rt.T
    return s, r, te - s * (r @ tt)


def write_splat(
    path: Path, xyz: np.ndarray, scale: np.ndarray, rgba: np.ndarray, quat: np.ndarray
) -> Path:
    """The 32-byte ``.splat`` record: f32 xyz, f32 scale (linear), u8 RGBA, u8 quaternion (w, x, y, z)."""
    rec = np.zeros(
        len(xyz), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)]
    )
    rec["p"], rec["s"], rec["c"] = xyz, scale, rgba
    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    rec["q"] = np.clip(np.round(q * 128 + 128), 0, 255)
    path.write_bytes(rec.tobytes())
    return path


def export_splat(src: str | Path, dst: Path, *, transform: tuple[float, np.ndarray, np.ndarray], max_splats: int = 1_500_000,
                 min_opacity: float = 0.02) -> dict:  # fmt: skip
    """Convert, move into the export frame, keep the ``max_splats`` most important; returns counts."""
    g = read_3dgs_ply(src)
    s, r, t = transform
    alpha = 1.0 / (1.0 + np.exp(-g["opacity"].astype(np.float64)))
    scale = np.exp(g["scale"].astype(np.float64)) * s
    keep = np.flatnonzero(alpha >= min_opacity)
    importance = alpha[keep] * np.prod(scale[keep], axis=1)
    keep = keep[np.argsort(-importance)[:max_splats]]  # most visible first
    xyz = (g["xyz"][keep].astype(np.float64) @ r.T) * s + t
    quat = _quat_mul(_quat_from_matrix(r), g["rot"][keep].astype(np.float64))
    rgb = np.clip(0.5 + SH_C0 * g["f_dc"][keep], 0.0, 1.0)
    rgba = np.concatenate([rgb, alpha[keep, None]], axis=1)
    write_splat(dst, xyz, scale[keep], np.round(rgba * 255).astype(np.uint8), quat)
    return {"splats": int(len(keep)), "trained": int(len(alpha)), "bytes": dst.stat().st_size}
