"""TSDF fusion in a child process: same surface as in-process, several models per worker, recovery from a dead worker."""

from __future__ import annotations

import numpy as np
import pytest

from drone3d.fastsfm.dense import Camera

o3d = pytest.importorskip("open3d")


def _plane_frames(n: int = 4) -> list:
    h, w, f = 120, 160, 150.0
    return [(np.full((h, w), 5.013, np.float32), np.full((h, w, 3), 128, np.uint8),
             Camera(f=f, cx=w / 2, cy=h / 2, k1=0.0, rotation=np.eye(3), translation=np.array([-0.1 * k, 0.0, 0.0])))
            for k in range(n)]  # fmt: skip


def test_worker_fuses_a_plane_twice_and_survives_a_crash() -> None:
    from drone3d.fastsfm.fusion_worker import FusionWorker

    kw = {"voxel": 0.05, "depth_max": 10.0, "trunc_voxels": 4.0, "memory_gb": 0.5}
    with FusionWorker() as fw:
        a = fw.fuse(_plane_frames(), **kw)
        assert len(a["triangles"]) > 100 and len(a["points"]) > 100
        np.testing.assert_allclose(
            np.median(a["vertices"][:, 2]), 5.0, atol=0.06
        )  # the plane, within a voxel
        b = fw.fuse(_plane_frames(), **kw)  # a second model in the same worker
        assert len(b["triangles"]) == len(a["triangles"])
        fw._proc.kill()  # the worker dies (as after a segfault): the next model gets a fresh one
        fw._proc.join()
        c = fw.fuse(_plane_frames(), **kw)
        assert len(c["triangles"]) == len(a["triangles"]) and fw.restarts == 1
