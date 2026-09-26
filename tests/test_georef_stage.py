"""End-to-end test of the georeferencing stage with injected GPS fixtures.

The bundled sample video has no GPS (F7), which blocks *scoring* criteria 2 and 6
— but it does not block testing the code path. These tests build a COLMAP text
model plus a GPS-tagged frame manifest and run `Pipeline._stage_georef` for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from drone3d.config import load_config
from drone3d.geo.enu import enu_to_geodetic
from drone3d.pipeline import Pipeline
from drone3d.types import PipelineResult
from drone3d.utils.ply import load_ply, write_ply

ORIGIN = {"lat0": 12.5, "lon0": 77.6, "alt0": 500.0}


def _build_run(tmp_path: Path, *, n: int = 6, noise_m: float = 0.0) -> Path:
    """Create a minimal run directory the georef stage can consume.

    Camera centres are placed on a known ENU layout and their GPS is derived
    through the real `enu_to_geodetic`, so the Umeyama fit has a well-defined
    ground truth. `noise_m` perturbs the GPS to check the reported RMSE.
    """
    run = tmp_path / "run"
    text_model = run / "sfm" / "sparse_txt"
    text_model.mkdir(parents=True)
    (run / "ingest").mkdir(parents=True)

    rng = np.random.default_rng(0)
    centers = rng.normal(scale=30.0, size=(n, 3))

    # images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # ImagePose.center == -R^T @ tvec, so with an identity rotation the camera
    # centre is -tvec.
    pose_lines = []
    for i, center in enumerate(centers):
        tx, ty, tz = (-float(c) for c in center)
        pose_lines.append(f"{i + 1} 1 0 0 0 {tx:.8f} {ty:.8f} {tz:.8f} 1 frame_{i:06d}.jpg")
        pose_lines.append("")  # blank points2D line
    (text_model / "images.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    csv_lines = ["index,timestamp_s,path,lat,lon,alt_m"]
    for i, center in enumerate(centers):
        noisy = center + rng.normal(scale=noise_m, size=3) if noise_m else center
        lat, lon, alt = enu_to_geodetic(float(noisy[0]), float(noisy[1]), float(noisy[2]), **ORIGIN)
        csv_lines.append(f"{i},{i}.0,frames/frame_{i:06d}.jpg,{lat:.8f},{lon:.8f},{alt:.3f}")
    (run / "ingest" / "frames.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # A minimal sparse cloud the stage can georeference.
    sparse_ply = run / "sfm" / "sparse.ply"
    write_ply(sparse_ply, np.arange(18, dtype=np.float64).reshape(6, 3))

    (run / "sfm" / "result.json").write_text(
        json.dumps(
            {
                "backend": "colmap",
                "sparse_ply": str(sparse_ply),
                "metadata": {"text_model": str(text_model)},
            }
        ),
        encoding="utf-8",
    )
    return run


def _pipeline(run: Path) -> Pipeline:
    config = load_config(None, ["geo.origin_lat=12.5", "geo.origin_lon=77.6", "geo.origin_alt=500"])
    pipeline = object.__new__(Pipeline)
    pipeline.run_dir = run
    pipeline.config = config
    pipeline._result = PipelineResult(run_dir=run)
    return pipeline


def test_georef_stage_runs_end_to_end(tmp_path: Path) -> None:
    run = _build_run(tmp_path, n=6)

    report = _pipeline(run)._stage_georef()

    assert report.status == "ok", report.message
    assert (run / "georef" / "georeferenced_sparse.ply").is_file()
    assert (run / "georef" / "camera_track.geojson").is_file()

    result = json.loads((run / "georef" / "result.json").read_text(encoding="utf-8"))
    assert result["num_correspondences"] == 6
    assert result["gps_accuracy"]["rmse_horizontal_m"] < 1.0


def test_georef_skips_without_gps(tmp_path: Path) -> None:
    """The real F7 situation: poses exist, no GPS anywhere."""
    run = _build_run(tmp_path, n=6)
    (run / "ingest" / "frames.csv").write_text(
        "index,timestamp_s,path,lat,lon,alt_m\n"
        + "\n".join(f"{i},{i}.0,frames/frame_{i:06d}.jpg,,," for i in range(6))
        + "\n",
        encoding="utf-8",
    )

    report = _pipeline(run)._stage_georef()

    assert report.status == "skipped"
    assert "GPS" in report.message or "telemetry" in report.message


def test_georef_skips_below_min_correspondences(tmp_path: Path) -> None:
    run = _build_run(tmp_path, n=2)

    report = _pipeline(run)._stage_georef()

    assert report.status == "skipped"
    assert "correspondence" in report.message


def test_georef_rmse_reflects_injected_gps_noise(tmp_path: Path) -> None:
    """Criterion 6's *measurement* must track the real error it reports."""
    quiet = (
        json.loads(
            (_build_run(tmp_path / "q", n=8) / "georef" / "result.json").read_text(encoding="utf-8")
        )
        if False
        else None
    )  # placeholder, computed below

    run_quiet = _build_run(tmp_path / "q", n=8, noise_m=0.0)
    _pipeline(run_quiet)._stage_georef()
    quiet = json.loads((run_quiet / "georef" / "result.json").read_text(encoding="utf-8"))

    run_noisy = _build_run(tmp_path / "n", n=8, noise_m=5.0)
    _pipeline(run_noisy)._stage_georef()
    noisy = json.loads((run_noisy / "georef" / "result.json").read_text(encoding="utf-8"))

    assert quiet["gps_accuracy"]["rmse_horizontal_m"] < 1.0
    assert noisy["gps_accuracy"]["rmse_horizontal_m"] > quiet["gps_accuracy"]["rmse_horizontal_m"]


def test_georeferenced_cloud_is_written_and_finite(tmp_path: Path) -> None:
    run = _build_run(tmp_path, n=6)
    sparse = run / "sfm" / "sparse.ply"
    write_ply(sparse, np.arange(18, dtype=np.float64).reshape(6, 3))

    report = _pipeline(run)._stage_georef()

    assert report.status == "ok", report.message
    out = run / "georef" / "georeferenced_sparse.ply"
    assert out.is_file()
    assert np.isfinite(load_ply(out).points).all()
