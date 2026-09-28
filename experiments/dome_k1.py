"""Does refining the radial distortion dome the reconstruction? Each pass re-mapped with k1 refined (as now) and fixed at 0.

    uv run python experiments/dome_k1.py RUN:PASS [RUN:PASS ...]   (writes paper/figures/dome_k1.json)

Dome index: the model levelled with the export's rule, the scene gridded 10 x 10, the 20th-percentile height of
the SfM points per cell taken as ground, a quadratic surface fitted to those; the rise of its quadratic part
across the scene, as a share of the flying height. A flat city gives ~0; a bowl or dome, its depth.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)
from drone3d.export.stage import _level

ROOT = Path(__file__).resolve().parents[1]


def dome_index(rec) -> dict:  # type: ignore[no-untyped-def]
    ims = [im for im in rec.images.values() if im.has_pose]
    cams = np.array([im.projection_center() for im in ims])
    rots = np.array([im.cam_from_world().rotation.matrix() for im in ims])
    pts = np.array([p.xyz for p in rec.points3D.values()])
    rot = _level(pts, cams, rots)
    p, c = pts @ rot.T, cams @ rot.T
    lo, hi = np.percentile(p[:, :2], 5, 0), np.percentile(p[:, :2], 95, 0)
    cells = []
    for i in range(10):
        for j in range(10):
            m = ((p[:, 0] >= lo[0] + i * (hi[0] - lo[0]) / 10) & (p[:, 0] < lo[0] + (i + 1) * (hi[0] - lo[0]) / 10)
                 & (p[:, 1] >= lo[1] + j * (hi[1] - lo[1]) / 10) & (p[:, 1] < lo[1] + (j + 1) * (hi[1] - lo[1]) / 10))  # fmt: skip
            if m.sum() >= 5:
                q = p[m]
                cells.append([*q[:, :2].mean(0), np.percentile(q[:, 2], 20)])
    g = np.array(cells)
    x, y = (g[:, 0] - g[:, 0].mean()) / ((hi - lo).max() / 2), (g[:, 1] - g[:, 1].mean()) / ((hi - lo).max() / 2)
    A = np.stack([x * x, y * y, x * y, x, y, np.ones_like(x)], 1)
    coef = np.linalg.lstsq(A, g[:, 2], rcond=None)[0]
    quad = A[:, :3] @ coef[:3]
    height = float(np.median(c[:, 2]) - np.median(g[:, 2]))
    return {"dome": round(float(np.ptp(quad) / max(height, 1e-9)), 4), "sign": "dome" if coef[0] + coef[1] < 0 else "bowl",
            "height": round(height, 3), "cells": len(g)}  # fmt: skip


def main() -> None:
    import pycolmap

    rows = []
    for arg in sys.argv[1:]:
        run, pass_name = arg.split(":")
        work = ROOT / "outputs" / run / "sfm" / "work"
        db, images = work / f"{pass_name}.db", ROOT / "outputs" / run / "dataset" / "images"
        for refine_k1 in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                opts = pycolmap.GlobalPipelineOptions()
                opts.mapper.keep_max_num_tracks = 1000
                opts.mapper.skip_retriangulation = True
                opts.mapper.bundle_adjustment.refine_extra_params = refine_k1
                shutil.copy(db, Path(tmp) / "db.db")
                if not refine_k1:  # start from k1 = 0 and keep it there
                    d = pycolmap.Database.open(str(Path(tmp) / "db.db"))
                    for cam in d.read_all_cameras():
                        cam.params = [*cam.params[:3], 0.0][: len(cam.params)]
                        d.update_camera(cam)
                    d.close()
                recs = pycolmap.global_mapping(str(Path(tmp) / "db.db"), str(images), str(Path(tmp) / "out"), opts)
                best = max(recs.values(), key=lambda r: r.num_reg_images())
                cam = next(iter(best.cameras.values()))
                row = {"run": run, "pass": pass_name, "refine_k1": refine_k1, "registered": best.num_reg_images(),
                       "focal": round(float(cam.params[0]), 1), "k1": round(float(cam.params[3]) if len(cam.params) > 3 else 0.0, 4),
                       "reproj_px": round(float(best.compute_mean_reprojection_error()), 3), **dome_index(best)}  # fmt: skip
                rows.append(row)
                print(row, flush=True)
    (ROOT / "paper" / "figures" / "dome_k1.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
