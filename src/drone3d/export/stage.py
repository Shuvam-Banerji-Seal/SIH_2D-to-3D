"""The ``export`` stage: deliverables in the problem statement's formats, and the web viewer.

Per dense model: georeferenced (ENU metres, z up) when the georef stage
solved it, otherwise the SfM frame levelled on its ground plane (arbitrary
units, z up). Writes mesh.{ply,obj,glb,fbx}, points.{ply,las},
dsm.tif + ortho.tif (UTM with an EPSG code when georeferenced), and a
self-contained web viewer (``index.html`` + vendored three.js) whose
``scene.json`` lists the models.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from drone3d.export.formats import rasterize_top, write_fbx, write_geotiff, write_las, write_mesh
from drone3d.logging_utils import get_logger

__all__ = ["run_export", "utm_epsg"]

log = get_logger(__name__)
VIEWER = Path(__file__).resolve().parents[1] / "viewer" / "static"


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _enu_to_utm(p: np.ndarray, origin: dict, epsg: int) -> np.ndarray:
    """Local ENU metres (tangent plane at ``origin``) -> UTM easting/northing/ellipsoidal height."""
    import pyproj

    lat0, lon0, h0 = origin["lat0"], origin["lon0"], origin.get("alt0", 0.0)
    to_ecef = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = to_ecef.transform(lon0, lat0, h0)
    la, lo = np.radians(lat0), np.radians(lon0)
    rot = np.array([[-np.sin(lo), np.cos(lo), 0],
                    [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                    [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])  # fmt: skip
    ecef = p @ rot + np.array([x0, y0, z0])
    to_utm = pyproj.Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    e, n, h = to_utm.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return np.stack([e, n, h], 1)


def _level(points: np.ndarray, cams: np.ndarray) -> np.ndarray:
    """Rotation taking the model's ground normal to +z (identity if it cannot be estimated)."""
    from drone3d.geo.georef import _rotation_between, estimate_up

    try:
        up, _ = estimate_up(points, cams)
    except Exception:  # too few points for a plane
        return np.eye(3)
    return _rotation_between(np.asarray(up, dtype=np.float64), np.array([0.0, 0.0, 1.0]))


def run_export(dense: dict, georef: dict | None, out_dir: Path, *, title: str, mesh_formats: list[str],
               las: bool = True, geotiff: bool = True, raster_cell: float | None = None, viewer: bool = True) -> dict:  # fmt: skip
    import open3d as o3d
    import pycolmap

    from drone3d.geo.georef import SimilarityTransform

    out_dir.mkdir(parents=True, exist_ok=True)
    geo_by_model = {m["model"]: m for m in (georef or {}).get("models", [])}
    origin = (georef or {}).get("origin")
    scene_models, rows = [], []
    for m in dense.get("models", []):
        if m.get("status") != "ok":
            continue
        name = Path(m["model"]).name
        mdir = out_dir / f"model_{name}"
        mdir.mkdir(parents=True, exist_ok=True)
        mesh = o3d.io.read_triangle_mesh(m["mesh"])
        pcd = o3d.io.read_point_cloud(m["points"])
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        vc = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
        p = np.asarray(pcd.points)
        pc = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        rec = pycolmap.Reconstruction(m["model"])
        posed = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if im.has_pose]
        cams = np.array([im.projection_center() for im in posed])
        # initial viewer pose: the middle keyframe's camera, looking at the model's median depth
        mid = posed[len(posed) // 2]
        axis = mid.cam_from_world().rotation.matrix()[2]  # optical axis in world coordinates
        depth_med = float(np.median((p - cams[len(posed) // 2]) @ axis)) if len(p) else 1.0
        view = np.array([cams[len(posed) // 2], cams[len(posed) // 2] + depth_med * axis])
        geo = geo_by_model.get(m["model"])
        if geo is not None:
            tr = geo["transform"]
            t = SimilarityTransform(scale=tr["scale"], rotation=np.asarray(tr["rotation"]), translation=np.asarray(tr["translation"]))
            v, p, cams, view, scale = t.apply(v), t.apply(p), t.apply(cams), t.apply(view), float(t.scale)
            units, frame = "m", "ENU"
        else:
            rot = _level(p, cams)
            v, p, cams, view, scale = v @ rot.T, p @ rot.T, cams @ rot.T, view @ rot.T, 1.0
            units, frame = "model units", "SfM (levelled, not georeferenced)"
        files = write_mesh(v, f, vc, mdir / "mesh", tuple(x for x in mesh_formats if x != "fbx"))
        if "fbx" in mesh_formats:
            fbx = write_fbx(mdir / "mesh.glb" if (mdir / "mesh.glb").is_file() else files[0], mdir / "mesh.fbx")
            files += [fbx] if fbx else []
        pts_ply = mdir / "points.ply"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        if pc is not None:
            cloud.colors = o3d.utility.Vector3dVector(pc / 255.0)
        o3d.io.write_point_cloud(str(pts_ply), cloud)
        files.append(pts_ply)
        epsg, p_proj = None, p
        if geo is not None and origin:
            epsg = utm_epsg(origin["lat0"], origin["lon0"])
            p_proj = _enu_to_utm(p, origin, epsg)
        if las:
            files.append(write_las(p_proj, pc, mdir / "points.las", epsg=epsg))
        if geotiff and len(p_proj):
            cell = raster_cell or 2.0 * m.get("voxel", 0.0) * scale or float(np.ptp(p_proj[:, :2], 0).max() / 1000)
            dsm, ortho, org = rasterize_top(p_proj, pc, cell)
            files.append(write_geotiff(dsm, mdir / "dsm.tif", origin=org, cell=cell, epsg=epsg))
            if ortho is not None:
                files.append(write_geotiff(ortho, mdir / "ortho.tif", origin=org, cell=cell, epsg=epsg, nodata=None))
        rel = [str(x.relative_to(out_dir)) for x in files if x]
        rows.append({"model": m["model"], "frame": frame, "units": units, "epsg": epsg, "files": rel,
                     "vertices": len(v), "triangles": len(f), "points": len(p)})  # fmt: skip
        scene_models.append({
            "name": f"model {name}", "mesh": f"model_{name}/mesh.glb" if "glb" in mesh_formats else None,
            "points": f"model_{name}/points.ply", "cameras": cams.round(4).tolist(), "units": units,
            "georeferenced": geo is not None, "up": [0, 0, 1],
            # frame the view on the bulk of the model, not on stray far-field fragments
            "view": {"eye": view[0].round(4).tolist(), "target": view[1].round(4).tolist()},
            "bounds": np.vstack([np.percentile(v if len(v) else p, [2, 98], axis=0), cams.min(0), cams.max(0)]).round(4).tolist() if len(cams) else None,
            "files": [{"label": Path(r).name, "path": r} for r in rel],
            "stats": {"keyframes": m.get("keyframes"), "triangles": f"{len(f):,}", "points": f"{len(p):,}",
                      "depth coverage": f"{100 * m.get('coverage', 0):.0f} %", "frame": frame, **({"EPSG": epsg} if epsg else {})},
        })  # fmt: skip
    if viewer and scene_models:
        for item in ("index.html", "vendor"):
            src, dst = VIEWER / item, out_dir / item
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        (out_dir / "scene.json").write_text(json.dumps({"title": title, "models": scene_models}, indent=1))
    return {"models": rows, "viewer": str(out_dir / "index.html") if viewer and scene_models else None}
