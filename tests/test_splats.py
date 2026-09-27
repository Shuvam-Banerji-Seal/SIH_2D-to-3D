"""3DGS PLY -> web .splat: values, importance order, pruning and the frame change."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drone3d.export.splats import SH_C0, _quat_from_matrix, compose, export_splat, read_3dgs_ply


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
    }[axis]


def _matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _write_ply(path: Path, xyz, f_dc, opacity, scale, rot) -> Path:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(9)],
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]  # fmt: skip
    n = len(xyz)
    data = np.zeros((n, len(names)), np.float32)
    data[:, 0:3], data[:, 6:9], data[:, 18] = xyz, f_dc, opacity
    data[:, 19:22], data[:, 22:26] = scale, rot
    head = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n" + "".join(
        f"property float {k}\n" for k in names
    )
    path.write_bytes((head + "end_header\n").encode() + data.tobytes())
    return path


def _read_splat(path: Path) -> np.ndarray:
    return np.frombuffer(
        path.read_bytes(), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)]
    )


def test_values_order_and_pruning(tmp_path: Path) -> None:
    xyz = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], np.float32)
    f_dc = np.array([[0, 0, 0], [1, -1, 0.5], [0, 0, 0]], np.float32)
    opacity = np.array([4.0, 4.0, -8.0], np.float32)  # sigmoid: 0.98, 0.98, 0.0003 (pruned)
    scale = np.log(np.array([[0.1, 0.1, 0.1], [1.0, 2.0, 0.5], [1, 1, 1]], np.float32))
    rot = np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1))
    src = _write_ply(tmp_path / "s.ply", xyz, f_dc, opacity, scale, rot)
    assert read_3dgs_ply(src)["xyz"].shape == (3, 3)

    info = export_splat(src, tmp_path / "s.splat", transform=(1.0, np.eye(3), np.zeros(3)))
    rec = _read_splat(tmp_path / "s.splat")
    assert info == {"splats": 2, "trained": 3, "bytes": 64}
    np.testing.assert_allclose(rec["p"][0], [1, 2, 3])  # the larger splat comes first
    np.testing.assert_allclose(rec["s"][0], [1.0, 2.0, 0.5], rtol=1e-6)
    expected = np.round(np.clip(0.5 + SH_C0 * f_dc[1], 0, 1) * 255)
    np.testing.assert_array_equal(rec["c"][0][:3], expected)
    assert rec["c"][0][3] == round(255 / (1 + np.exp(-4.0)))
    np.testing.assert_array_equal(
        rec["q"][0], [255, 128, 128, 128]
    )  # identity (w=1 -> 128 + 128, clipped)


def test_frame_change_moves_positions_rotations_and_scales(tmp_path: Path) -> None:
    q0 = _quat_from_matrix(_rot("x", 30))
    src = _write_ply(tmp_path / "s.ply", np.array([[1, 0, 0]], np.float32), np.zeros((1, 3), np.float32),
                     np.array([5.0], np.float32), np.log(np.array([[0.2, 0.4, 0.8]], np.float32)),
                     q0[None].astype(np.float32))  # fmt: skip
    # trained in a frame scaled by 2 and shifted; exported into a frame rotated 90 deg about z, scaled 3
    train = (2.0, np.eye(3), np.array([1.0, 0, 0]))
    export = (3.0, _rot("z", 90), np.array([0, 0, 10.0]))
    s, r, t = compose(export, train)
    export_splat(src, tmp_path / "s.splat", transform=(s, r, t))
    rec = _read_splat(tmp_path / "s.splat")
    world = (np.array([1.0, 0, 0]) - train[2]) / train[0]  # train -> world: (1,0,0) -> (0,0,0)
    np.testing.assert_allclose(rec["p"][0], export[0] * export[1] @ world + export[2], atol=1e-6)
    np.testing.assert_allclose(rec["s"][0], np.array([0.2, 0.4, 0.8]) * 3.0 / 2.0, rtol=1e-6)
    got = _matrix((rec["q"][0].astype(float) - 128) / 128)
    np.testing.assert_allclose(got, _rot("z", 90) @ _rot("x", 30), atol=0.03)  # 8-bit quaternion
