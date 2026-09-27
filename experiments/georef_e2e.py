"""End-to-end check of the georeferenced fast profile with synthetic GPS on a real run.

None of the sample videos carries a flight log, so the georef -> export path
(ENU metres, UTM with an EPSG code in LAS and GeoTIFF) is exercised here: each
SfM model of a finished run gets a known model-to-world similarity (levelled on
its ground plane, random yaw, a scale giving a ~250 m track, placed at the real
site), its keyframes get GPS = that transform applied to their camera centres +
consumer-grade noise, and the pipeline's georef, export, metrics and report
stages run on a copy of the run. Reported: the error of the exported mesh
vertices against the truth (the problem statement's "spatial accuracy <= 1 m"),
of the camera track, the metric scale, and the CRS tags of the LAS / GeoTIFF.

    uv run python experiments/georef_e2e.py outputs/jal_mahal_rel6 26.9535 75.8462
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)


def main() -> None:
    import laspy
    import open3d as o3d
    import pycolmap
    import tifffile

    from drone3d.export.stage import _level
    from drone3d.geo.projection import LocalTangentPlane

    src = Path(sys.argv[1])
    lat0, lon0 = float(sys.argv[2]), float(sys.argv[3])
    noise_h, noise_v = 1.5, 3.0
    run = ROOT / "outputs" / "experiments" / f"georef_e2e_{src.name}"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    for stage in ("ingest", "keyframes", "sfm", "dense"):
        shutil.copytree(src / stage, run / stage)
    (run / "dataset").symlink_to((src / "dataset").resolve())
    rng = np.random.default_rng(7)
    plane = LocalTangentPlane(lat0, lon0, 450.0)
    rows = json.loads((run / "keyframes" / "keyframes.json").read_text())
    by_name = {r["name"]: r for r in rows}
    sfm = json.loads((run / "sfm" / "result.json").read_text())
    truth = {}
    for k, m in enumerate(sfm["models"]):
        rec = pycolmap.Reconstruction(m["path"])
        ims = [im for im in rec.images.values() if im.has_pose]
        cams = np.array([im.projection_center() for im in ims])
        pts = np.array([p.xyz for p in rec.points3D.values()])
        level = _level(pts, cams)
        yaw = rng.uniform(0, 2 * np.pi)
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rot = rz @ level
        extent = float(np.linalg.norm(np.ptp(cams, axis=0))) or 1.0
        scale = 250.0 / extent
        offset = np.array([400.0 * k, -300.0 * k, 100.0]) + rng.normal(0, 50, 3) * [1, 1, 0]
        truth[m["path"]] = (scale, rot, offset)
        world = scale * cams @ rot.T + offset
        for im, w in zip(ims, world, strict=True):
            noisy = w + np.r_[rng.normal(0, noise_h, 2), rng.normal(0, noise_v)]
            lat, lon, alt = plane.to_geodetic(noisy)
            by_name[im.name].update(lat=lat, lon=lon, alt_m=alt)
    (run / "keyframes" / "keyframes.json").write_text(json.dumps(rows, indent=1))
    # the truth's ENU origin, so exported ENU coordinates are directly comparable
    cmd = [str(Path(sys.executable).with_name("drone3d")), "run", "--config", str(ROOT / "configs" / "fast.yaml"),
           "--set", f"run_name={run.name}", "--set", f"geo.origin_lat={lat0}", "--set", f"geo.origin_lon={lon0}",
           "--set", "geo.origin_alt=450.0", "--run-dir", str(run), "--stages", "georef,export,metrics,report"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    print("\n".join(line for line in res.stderr.splitlines() if "stage " in line and ": " in line)[-2000:])
    geo = json.loads((run / "georef" / "result.json").read_text())
    exp = json.loads((run / "export" / "result.json").read_text())
    report = {"gps_noise_m": {"horizontal": noise_h, "vertical": noise_v}, "models": []}
    for g in geo["models"]:
        scale, rot, offset = truth[g["model"]]
        dense = next((d for d in json.loads((run / "dense" / "result.json").read_text())["models"] if d["model"] == g["model"]), None)
        e = next((x for x in exp["models"] if x["model"] == g["model"]), None)
        r_est = np.asarray(g["transform"]["rotation"])
        d = r_est @ rot.T
        rot_err = float(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
        tilt = float(np.degrees(np.arccos(np.clip((r_est @ np.linalg.inv(rot) @ [0, 0, 1])[2], -1, 1))))
        row = {"model": g["model"], "mode": g.get("mode"), "held_out": g.get("held_out"),
               "scale_true": round(scale, 4), "scale_est": round(float(g["transform"]["scale"]), 4),
               "rotation_err_deg": round(rot_err, 3), "tilt_err_deg": round(tilt, 3)}  # fmt: skip
        if dense and e:
            mesh = o3d.io.read_triangle_mesh(dense["mesh"])
            v_model = np.asarray(mesh.vertices)
            v_true = scale * v_model @ rot.T + offset
            # the georef stage puts its ENU origin at the median GPS altitude, not at the truth's 450 m
            v_true[:, 2] -= geo["origin"]["alt0"] - 450.0
            out = o3d.io.read_triangle_mesh(str(run / "export" / f"model_{Path(g['model']).name}" / "mesh.ply"))
            v_est = np.asarray(out.vertices)
            err = np.linalg.norm(v_est - v_true, axis=1)
            h_err = np.linalg.norm((v_est - v_true)[:, :2], axis=1)
            rec = pycolmap.Reconstruction(g["model"])
            track = scale * np.array([im.projection_center() for im in rec.images.values() if im.has_pose]) @ rot.T + offset
            dist = np.min(np.linalg.norm(v_true[:, None, :2] - track[None, ::max(1, len(track) // 50), :2], axis=2), axis=1)
            bands = {}
            for lo, hi in ((0, 100), (100, 300), (300, 1e9)):
                sel = (dist >= lo) & (dist < hi)
                if sel.any():
                    bands[f"{lo}-{hi if hi < 1e9 else 'inf'} m"] = {"share": round(float(sel.mean()), 3),
                                                                  "median": round(float(np.median(err[sel])), 2),
                                                                  "p90": round(float(np.percentile(err[sel], 90)), 2)}  # fmt: skip
            row.update(mesh_error_m={"median": round(float(np.median(err)), 3), "p95": round(float(np.percentile(err, 95)), 3),
                                     "horizontal_median": round(float(np.median(h_err)), 3), "by_distance_from_track": bands},
                       epsg=e.get("epsg"))  # fmt: skip
            mdir = run / "export" / f"model_{Path(g['model']).name}"
            las = laspy.read(str(mdir / "points.las"))
            row["las_epsg"] = las.header.parse_crs().to_epsg() if las.header.parse_crs() else None
            row["las_first_point"] = [round(float(las.x[0]), 1), round(float(las.y[0]), 1), round(float(las.z[0]), 1)]
            with tifffile.TiffFile(mdir / "dsm.tif") as tif:
                row["dsm_epsg"] = int(tif.geotiff_metadata.get("ProjectedCSTypeGeoKey", 0))
        report["models"].append(row)
        print(json.dumps(row), flush=True)
    (run / "georef_e2e.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
