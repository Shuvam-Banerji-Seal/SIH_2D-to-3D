"""The ``export`` stage: deliverables in the problem statement's formats, and the web viewer.

Per dense model: georeferenced (ENU metres, z up) when the georef stage
solved it, otherwise the SfM frame levelled on its ground plane (arbitrary
units, z up). Writes mesh.{ply,obj,glb,fbx}, points.{ply,las},
dsm.tif + ortho.tif (UTM with an EPSG code when georeferenced), and a
self-contained web viewer (``index.html`` + vendored three.js) whose
``scene.json`` lists the models.
"""

from __future__ import annotations

import collections
import contextlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from drone3d.export.formats import (
    Y_UP,
    rasterize_top,
    write_blend,
    write_fbx,
    write_geotiff,
    write_las,
    write_mesh,
)
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


class _Clock:
    """Seconds per named step, summed over models: ``with clock("texture"): ...``."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = collections.defaultdict(float)

    @contextlib.contextmanager
    def __call__(self, key: str):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[key] += time.perf_counter() - t0

    def rounded(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.seconds.items()}


def _axis_depth(points: np.ndarray, eye: np.ndarray, axis: np.ndarray, cone_deg: float = 3.0) -> float:
    """How far along ``axis`` from ``eye`` the surface is: the viewer's orbit pivot.

    The median depth of the points inside a narrow cone around the axis, where the
    drone actually looked; the median over the whole cloud put Colosseum's pivot 19
    model units below its ground (the far field pulls it out), so orbiting swung
    the model around a point hidden beneath it.
    """
    if not len(points):
        return 1.0
    rel = np.asarray(points, dtype=np.float64) - eye
    along = rel @ axis
    perp = np.linalg.norm(rel - along[:, None] * axis, axis=1)
    for widen in (1, 3, 10):  # a sparse cloud: widen the cone before falling back to every point
        if (cone := (along > 0) & (perp < np.tan(np.radians(cone_deg * widen)) * along)).sum() >= 50:
            return float(np.median(along[cone]))
    return float(np.median(along[along > 0])) if (along > 0).any() else 1.0


def _subject(cams: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, float] | None:
    """What the flight looks at -> ``(point, radius)``, or None when there is no such thing.

    The point is closest, in least squares, to every keyframe's optical axis; the radius is the cameras'
    median distance to it. A nadir survey or a straight fly-by (near-parallel axes) has no such point, and a
    pan or a turning fly-by has one nobody looks at. On the fifteen sample videos' 95 models, 42 had a point;
    7 of them were aimed at by no keyframe at all (Hanoi, Eiffel Tower, Qutub Minar, Reichstag, Cristo
    Redentor) and 4 by under 30 % (the FPV flights). Asking at least 30 % of the keyframes to aim within
    25 degrees of it keeps 31 (experiments/generality.py).
    """
    if len(cams) < 3:
        return None
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(cams, axes, strict=True):
        proj = np.eye(3) - np.outer(d, d)
        a += proj
        b += proj @ c
    w = np.linalg.eigvalsh(a)
    if w[0] <= 0.02 * w[-1]:
        return None
    point = np.linalg.solve(a, b)
    rel = point - cams
    dist = np.linalg.norm(rel, axis=1)
    aimed = np.einsum("ij,ij->i", rel, axes) / np.maximum(dist, 1e-12) > np.cos(np.radians(25))
    if aimed.mean() < 0.3:
        return None
    return point, float(np.median(dist))


def _subject_view(points: np.ndarray, cams: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """The viewer's first pose ``[eye, target]``: the keyframe that looks most directly at what the drone circled.

    Of the keyframes, the one whose axis passes nearest the subject (in angle) is the eye, looking at its own
    axis depth. The middle keyframe, the previous choice, looked at Rome's skyline in the Colosseum model
    merged from seven shots. When the axes hardly converge (a nadir survey, a straight fly-by), the middle
    keyframe stays.
    """
    pick = len(cams) // 2
    sub = _subject(cams, axes)
    if sub is not None:
        rel = sub[0] - cams
        cos = np.einsum("ij,ij->i", rel, axes) / np.maximum(np.linalg.norm(rel, axis=1), 1e-12)
        if cos.max() > np.cos(np.radians(20)):
            pick = int(np.argmax(cos))
    return np.array([cams[pick], cams[pick] + _axis_depth(points, cams[pick], axes[pick]) * axes[pick]])


def _clean_mesh(v: np.ndarray, f: np.ndarray, vc: np.ndarray | None, geo: dict | None, points: np.ndarray, cams: np.ndarray,
                cam_rots: np.ndarray, *, voxel: float, subject) -> tuple:  # type: ignore[no-untyped-def]  # fmt: skip
    """The clean model (drone3d.export.terrain) of a fused mesh, in the SfM frame the texture is baked in.

    Cleaning needs z up: the export's own frame (the georeference, or the levelling), and back. The opening
    window is 0.6 x the subject radius -- the cameras' distance to what they circle, wider than it -- or a tenth
    of the scene where the flight circles nothing. Colours: the nearest fused vertex's.
    """
    from scipy.spatial import cKDTree

    from drone3d.export.terrain import clean_model

    if geo is not None:
        tr = geo["transform"]
        s_, r_, t_ = float(tr["scale"]), np.asarray(tr["rotation"]), np.asarray(tr["translation"])
    else:
        s_, r_, t_ = 1.0, _level(points, cams, np.transpose(cam_rots, (0, 2, 1))), np.zeros(3)
    up = s_ * v @ r_.T + t_
    extent = float(np.linalg.norm(np.percentile(up[:, :2], 95, 0) - np.percentile(up[:, :2], 5, 0)))
    window = 0.6 * s_ * subject[1] if subject is not None else 0.1 * extent
    cv, cf, info = clean_model(up, f, voxel=max(voxel * s_, 1e-6), window=window)
    back = ((cv - t_) / s_) @ r_
    colours = None
    if vc is not None:
        _, near = cKDTree(v).query(back, workers=8)
        colours = vc[near]
    return back, cf, colours, info


def _level(points: np.ndarray, cams: np.ndarray, rotations: np.ndarray | None = None) -> np.ndarray:
    """Rotation taking the model's ground normal to +z (identity if it cannot be estimated)."""
    from drone3d.geo.georef import _rotation_between, estimate_up

    try:
        up, _ = estimate_up(points, cams, rotations=rotations)
    except Exception:  # too few points for a plane
        return np.eye(3)
    return _rotation_between(np.asarray(up, dtype=np.float64), np.array([0.0, 0.0, 1.0]))


def bake_texture(v: np.ndarray, f: np.ndarray, vc: np.ndarray | None, posed, rec, images: Path, *,
                 views: int = 48, size: int = 4096, gain: bool = True):  # type: ignore[no-untyped-def]  # fmt: skip
    """Keyframe texture for the mesh (model frame) -> ``(corner_uv, albedo, info)`` or ``None``."""
    import torch
    from torchvision.io import read_file

    from drone3d.export.texture_gpu import View, bake_soup_texture
    from drone3d.gpu.nvjpeg import decode_jpeg

    if not len(f) or not torch.cuda.is_available():
        return None
    pick = [posed[int(round(i))] for i in np.linspace(0, len(posed) - 1, min(views, len(posed)))]
    from drone3d.io.overlay import load_mask

    overlay = load_mask(images.parent)  # the baker scales it to each view; None: the video has no overlay
    overlay = torch.from_numpy(overlay).cuda() if overlay is not None else None
    vs = []
    for im in pick:
        path = images / im.name
        if not path.is_file():
            continue
        img = decode_jpeg(read_file(str(path)), device="cuda").permute(1, 2, 0).contiguous()
        cam = rec.cameras[im.camera_id]
        s = img.shape[1] / cam.width  # keyframes may be stored smaller than the camera
        pose = im.cam_from_world()
        vs.append(View(cam.params[0] * s, cam.params[1] * s, cam.params[2] * s,
                       np.asarray(pose.rotation.matrix()), np.asarray(pose.translation), img,
                       float(cam.params[3]) if cam.model.name in ("SIMPLE_RADIAL", "RADIAL") else 0.0, overlay))  # fmt: skip
    if not vs:
        return None
    uv, albedo, info = bake_soup_texture(v, f, vs, size=size, fallback_rgb=vc, gain=gain)
    del vs
    torch.cuda.empty_cache()
    return uv, albedo, info


def _write_textured(v: np.ndarray, f: np.ndarray, uv: np.ndarray, albedo: np.ndarray, stem: Path) -> list[Path]:
    """Textured GLB and OBJ (+ .mtl, .jpg), the atlas stored as JPEG.

    PNG-encoding a 4096^2 atlas twice took 9 s per model; JPEG is ~20x faster
    and 5x smaller. The OBJ shares vertices and indexes texture coordinates
    separately (``f v/vt``) instead of tripling the vertex count.
    """
    import io

    import trimesh
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(albedo).save(buf, format="JPEG", quality=90)
    jpg = buf.getvalue()
    tex = stem.with_name(stem.name + "_albedo.jpg")
    tex.write_bytes(jpg)
    image = Image.open(io.BytesIO(jpg))  # format JPEG: trimesh embeds it as is in the GLB
    verts = v[f].reshape(-1, 3) @ Y_UP.T  # glTF is y-up
    faces = np.arange(len(verts)).reshape(-1, 3)
    visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), image=image)
    glb = stem.with_suffix(".glb")
    trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False).export(glb)
    obj, mtl = stem.with_suffix(".obj"), stem.with_suffix(".mtl")
    mtl.write_text(f"newmtl albedo\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd {tex.name}\n")
    n = len(f)
    fi = np.empty((n, 6), dtype=np.int64)
    fi[:, 0::2] = f + 1
    fi[:, 1::2] = np.arange(3 * n).reshape(n, 3) + 1
    with obj.open("w") as fh:
        fh.write(f"mtllib {mtl.name}\nusemtl albedo\n")
        _write_rows(fh, "v %.5f %.5f %.5f\n", v)
        _write_rows(fh, "vt %.6f %.6f\n", uv.reshape(-1, 2))
        _write_rows(fh, "f %d/%d %d/%d %d/%d\n", fi)
    return [glb, obj, mtl, tex]


def _decimate(mesh, v: np.ndarray, f: np.ndarray, vc: np.ndarray | None, target: int):  # type: ignore[no-untyped-def]
    """Quadric decimation to ``target`` triangles -> ``(vertices, faces, colours)``.

    fast-simplification (C++ quadric collapses) took 8.4 s for Petronas's
    3.7M-triangle model where Open3D took 56 s, and stayed as close to the
    original surface (median vertex offset 0.024 vs 0.028 model units). Each
    kept vertex takes the mean colour of its 4 nearest original vertices:
    replaying the collapses to average exactly the merged ones cost 31 s on a
    3.3M-triangle model, the k-d tree 1.5 s, and the colours differ by 0.3/255
    (median; 3.6 at the 90th percentile).
    """
    try:
        import fast_simplification as fs
    except ImportError:
        small = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        colours = (np.asarray(small.vertex_colors) * 255).astype(np.uint8) if small.has_vertex_colors() else None
        return np.asarray(small.vertices), np.asarray(small.triangles), colours
    dv, df = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1.0 - target / len(f))
    colours = None
    if vc is not None:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(v).query(dv, k=4, workers=8)
        colours = np.round(vc[:, :3][idx].astype(np.float64).mean(1)).astype(np.uint8)
    return dv.astype(np.float64), df.astype(np.int64), colours


def _frame_previews(posed: list, images: Path | None, depth_dir: Path | None, out: Path, *, most: int = 36,
                    width: int = 320) -> dict | None:  # fmt: skip
    """Small keyframe photos and fused-depth previews for the viewer's photo and depth layers."""
    if images is None or not posed:
        return None
    from PIL import Image

    step = max(1, -(-len(posed) // most))
    names = []
    for im in posed[::step]:
        src = images / im.name
        if not src.is_file():
            continue
        stem = Path(im.name).stem
        (out / "photo").mkdir(parents=True, exist_ok=True)
        with Image.open(src) as pic:
            pic.draft("RGB", (width, width))  # decode at 1/2..1/8 scale: the preview needs no more
            pic.thumbnail((width, width))
            pic.convert("RGB").save(out / "photo" / f"{stem}.jpg", "JPEG", quality=80)
        if depth_dir is not None and (depth_dir / f"{stem}.jpg").is_file():
            (out / "depth").mkdir(parents=True, exist_ok=True)
            shutil.copy2(depth_dir / f"{stem}.jpg", out / "depth" / f"{stem}.jpg")
        names.append(im.name)
    return {"names": names, "photo": (out / "photo").exists(), "depth": (out / "depth").exists()} if names else None


def _textured_and_fbx(baked: tuple | None, mdir: Path, fbx_src: Path | None, fbx: bool, blend: bool = False) -> list[Path]:
    """Background half of a model's export: textured GLB/OBJ, then FBX and .blend (from the textured GLB)."""
    out = _write_textured(*baked, mdir / "mesh_textured") if baked is not None else []
    src = out[0] if out else fbx_src
    if fbx and src is not None:
        converted = write_fbx(src, mdir / "mesh.fbx")
        if converted is not None:
            out.append(converted)
    if blend and src is not None and src.suffix == ".glb":
        scene = write_blend(src, mdir / "mesh.blend")
        if scene is not None:
            out.append(scene)
    return out


def _write_rows(fh, fmt: str, rows: np.ndarray, chunk: int = 200_000) -> None:  # type: ignore[no-untyped-def]
    """Text rows via one C-level ``%`` per chunk: ``np.savetxt`` formats row by row in
    Python and took 22 s for four ~1M-row OBJ files."""
    flat = rows.tolist() if rows.dtype.kind in "iu" else rows.astype(np.float64)
    for s in range(0, len(rows), chunk):
        block = flat[s : s + chunk]
        vals = [x for r in block for x in r] if isinstance(block, list) else block.ravel().tolist()
        fh.write((fmt * (len(vals) // fmt.count("%"))) % tuple(vals))


def run_export(dense: dict, georef: dict | None, out_dir: Path, *, title: str, mesh_formats: list[str],
               las: bool = True, geotiff: bool = True, raster_cell: float | None = None, viewer: bool = True,
               images: Path | None = None, texture: bool = True, texture_views: int = 48,
               texture_size: int = 4096, max_triangles: int = 600_000, splats: dict | None = None,
               max_splats: int = 1_500_000, texture_gain: bool = True, clean_mesh: bool = True) -> dict:  # fmt: skip
    import open3d as o3d
    import pycolmap

    from drone3d.geo.georef import SimilarityTransform

    out_dir.mkdir(parents=True, exist_ok=True)
    geo_by_model = {m["model"]: m for m in (georef or {}).get("models", [])}
    splat_by_model = {m["model"]: m for m in (splats or {}).get("models", []) if m.get("splat_ply")}
    origin = (georef or {}).get("origin")
    scene_models, rows = [], []
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=2)
    bg_jobs: list = []
    frames_jobs: list = []
    clock = _Clock()
    for m in dense.get("models", []):
        if m.get("status") != "ok":
            continue
        name = Path(m["model"]).name
        mdir = out_dir / f"model_{name}"
        if mdir.exists():  # our own output: stale files from an earlier export would be listed as current
            shutil.rmtree(mdir)
        mdir.mkdir(parents=True)
        with clock("read"):
            mesh = o3d.io.read_triangle_mesh(m["mesh"])
            pcd = o3d.io.read_point_cloud(m["points"])
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        vc = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
        p = np.asarray(pcd.points)
        pc = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        if not len(p) or not len(f):  # nothing to deliver (or a cloud without surface): skip, do not fail
            log.warning("export: model %s has %d points and %d triangles; skipped", name, len(p), len(f))
            continue
        rec = pycolmap.Reconstruction(m["model"])
        posed = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if im.has_pose]
        cams = np.array([im.projection_center() for im in posed])
        # world-from-camera rotations: the viewer flies the drone's path and looks through its keyframes
        cam_rots = np.array([im.cam_from_world().rotation.matrix().T for im in posed])
        cam0 = rec.cameras[posed[0].camera_id]
        intr = {"f": float(cam0.focal_length_x), "width": int(cam0.width), "height": int(cam0.height)}
        # initial viewer pose: the keyframe looking most directly at the subject (optical axes: third rows)
        axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in posed])
        view = _subject_view(p, cams, axes)
        subject = _subject(cams, axes)  # the viewer's focus box: the subject, a camera distance around it
        if subject is not None:
            view = np.vstack([view, subject[0]])  # moved into the export frame with the view
        # The full-density mesh is the measurement deliverable (PLY); the viewable copies
        # (GLB, textured OBJ/GLB, FBX) are capped: a 1.5M-triangle model made a 120 MB
        # GLB that browsers load slowly and a 4096^2 soup atlas cannot texture finely.
        # They are the clean model (drone3d.export.terrain): the ground as a terrain surface, what stands on it
        # smoothed, fragments dropped -- the fused mesh's crumpled walls and holed ground read as broken triangles.
        src_v, src_f, src_vc, src_mesh = v, f, vc, mesh
        clean_info = None
        if clean_mesh:
            with clock("clean"):
                try:
                    src_v, src_f, src_vc, clean_info = _clean_mesh(v, f, vc, geo_by_model.get(m["model"]), p, cams, cam_rots,
                                                                   voxel=float(m.get("voxel") or 0.0), subject=subject)  # fmt: skip
                    src_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(src_v), o3d.utility.Vector3iVector(src_f))
                    if src_vc is not None:
                        src_mesh.vertex_colors = o3d.utility.Vector3dVector(src_vc[:, :3] / 255.0)
                except (ValueError, IndexError, RuntimeError, MemoryError) as exc:  # an odd model: the fused mesh is still one
                    log.warning("clean model failed for model %s: %s", name, exc)
                    clean_info = {"status": "failed", "reason": str(exc)[:200]}
        dv, df, dvc = src_v, src_f, src_vc
        if len(src_f) > 1.25 * max_triangles:  # 630k -> 600k cost 2.3 s per model for nothing a viewer notices
            with clock("decimate"):
                dv, df, dvc = _decimate(src_mesh, src_v, src_f, src_vc, max_triangles)
        baked, tex_info = None, None
        if texture and images is not None:
            try:
                with clock("texture"):
                    from drone3d.engine.models import gpu_exclusive

                    with gpu_exclusive():
                        res = bake_texture(dv, df, dvc, posed, rec, images, views=texture_views, size=texture_size,
                                       gain=texture_gain)  # fmt: skip
                if res is not None:
                    baked, tex_info = (dv, df, res[0], res[1]), res[2]
            except (RuntimeError, ValueError) as exc:  # texturing improves the mesh; never lose the mesh over it
                log.warning("texture baking failed for model %s: %s", name, exc)
        geo = geo_by_model.get(m["model"])
        if geo is not None:
            tr = geo["transform"]
            t = SimilarityTransform(scale=tr["scale"], rotation=np.asarray(tr["rotation"]), translation=np.asarray(tr["translation"]))
            v, p, cams, view, scale = t.apply(v), t.apply(p), t.apply(cams), t.apply(view), float(t.scale)
            cam_rots = np.asarray(t.rotation) @ cam_rots
            dv = t.apply(dv)
            if baked is not None:
                baked = (t.apply(baked[0]), *baked[1:])
            units, frame = "m", "ENU"
            to_export = {"scale": float(t.scale), "rotation": np.asarray(t.rotation).tolist(), "translation": np.asarray(t.translation).tolist()}
        else:
            with clock("level"):
                rot = _level(p, cams, np.transpose(cam_rots, (0, 2, 1)))  # world-to-camera, as estimate_up takes them
            v, p, cams, view, scale = v @ rot.T, p @ rot.T, cams @ rot.T, view @ rot.T, 1.0
            cam_rots = rot @ cam_rots
            dv = dv @ rot.T
            if baked is not None:
                baked = (baked[0] @ rot.T, *baked[1:])
            units, frame = "model units", "SfM (levelled, not georeferenced)"
            to_export = {"scale": 1.0, "rotation": rot.tolist(), "translation": [0.0, 0.0, 0.0]}
        # keyframe / depth previews for the viewer's image layers: CPU work, in the background
        frames_job = pool.submit(_frame_previews, posed, images, Path(m["depth_previews"]) if m.get("depth_previews") else None,
                                 mdir / "frames")  # fmt: skip
        frames_jobs.append((frames_job, len(rows), name))
        # SfM frame -> this model's export frame, for renderers that follow the keyframe cameras
        (mdir / "frame.json").write_text(json.dumps({**to_export, "frame": frame, "units": units}, indent=1))
        splat_file, splat_info = None, None
        trained = splat_by_model.get(m["model"])
        if trained is not None and Path(trained["splat_ply"]).is_file():
            from drone3d.export.splats import compose, export_splat, train_from_world

            with clock("splats"):
                tf = compose((to_export["scale"], np.asarray(to_export["rotation"]), np.asarray(to_export["translation"])),
                             train_from_world(Path(trained["run_dir"]) / "scene_transform.json"))  # fmt: skip
                splat_file = mdir / "splats.splat"
                splat_info = export_splat(trained["splat_ply"], splat_file, transform=tf, max_splats=max_splats)
        # OBJ has no standard vertex colour: with a texture, OBJ is written textured only
        plain = [x for x in mesh_formats if x not in ("fbx", "blend") and not (x == "obj" and baked is not None)]
        with clock("mesh_files"):
            files = write_mesh(v, f, vc, mdir / "mesh", tuple(x for x in plain if x == "ply"))
            files += write_mesh(dv, df, dvc, mdir / "mesh", tuple(x for x in plain if x != "ply"))
        # textured files and the FBX are written in the background while the GPU bakes the next model
        fbx_src = mdir / "mesh.glb" if (mdir / "mesh.glb").is_file() else (files[0] if files else None)
        bg_jobs.append((pool.submit(_textured_and_fbx, baked, mdir, fbx_src, "fbx" in mesh_formats, "blend" in mesh_formats),
                        len(rows)))  # fmt: skip
        pts_ply = mdir / "points.ply"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        if pc is not None:
            cloud.colors = o3d.utility.Vector3dVector(pc / 255.0)
        with clock("points_ply"):
            o3d.io.write_point_cloud(str(pts_ply), cloud)
        files.append(pts_ply)
        epsg, p_proj = None, p
        if geo is not None and origin:
            epsg = utm_epsg(origin["lat0"], origin["lon0"])
            p_proj = _enu_to_utm(p, origin, epsg)
        if las:
            with clock("las"):
                files.append(write_las(p_proj, pc, mdir / "points.las", epsg=epsg))
        if geotiff and len(p_proj):
            with clock("geotiff"):
                cell = raster_cell or 2.0 * m.get("voxel", 0.0) * scale or float(np.ptp(p_proj[:, :2], 0).max() / 1000)
                dsm, ortho, org = rasterize_top(p_proj, pc, cell)
                files.append(write_geotiff(dsm, mdir / "dsm.tif", origin=org, cell=cell, epsg=epsg))
                if ortho is not None:
                    files.append(write_geotiff(ortho, mdir / "ortho.tif", origin=org, cell=cell, epsg=epsg,
                                               nodata=None))  # fmt: skip
        if splat_file is not None:
            files.append(splat_file)
        rel = [str(x.relative_to(out_dir)) for x in files if x]
        rows.append({"model": m["model"], "frame": frame, "units": units, "epsg": epsg, "files": rel,
                     "vertices": len(v), "triangles": len(f), "viewer_triangles": len(df), "points": len(p),
                     "texture": tex_info, "splats": splat_info, "clean": clean_info})  # fmt: skip
        scene_models.append({
            "name": f"model {name}",
            "mesh": f"model_{name}/mesh_textured.glb" if baked is not None else (f"model_{name}/mesh.glb" if "glb" in mesh_formats else None),
            "points": f"model_{name}/points.ply", "cameras": cams.round(4).tolist(), "units": units, "dir": f"model_{name}", "gltf_up": "y",
            "frames": None,  # filled in when the background previews finish
            "camera_rotations": cam_rots.reshape(len(cam_rots), 9).round(5).tolist(), "intrinsics": intr,
            "images": [im.name for im in posed],
            "splat": f"model_{name}/splats.splat" if splat_file is not None else None,
            "georeferenced": geo is not None, "up": [0, 0, 1],
            # frame the view on the bulk of the model, not on stray far-field fragments
            "view": {"eye": view[0].round(4).tolist(), "target": view[1].round(4).tolist()},
            "focus": {"center": view[2].round(4).tolist(), "radius": round(subject[1] * scale, 4)} if subject is not None else None,
            "bounds": np.vstack([np.percentile(v if len(v) else p, [2, 98], axis=0), cams.min(0), cams.max(0)]).round(4).tolist() if len(cams) else None,
            "files": [{"label": Path(r).name, "path": r} for r in rel],
            "stats": {"keyframes": m.get("keyframes"), "triangles": f"{len(f):,}", "points": f"{len(p):,}",
                      "depth coverage": f"{100 * m.get('coverage', 0):.0f} %", "frame": frame, **({"EPSG": epsg} if epsg else {})},
        })  # fmt: skip
    for fut, k, name in frames_jobs:
        with clock("frames_wait"):
            fr = fut.result()
        if fr:
            scene_models[k]["frames"] = {**fr, "photo_dir": f"model_{name}/frames/photo", "depth_dir": f"model_{name}/frames/depth"}
    for fut, k in bg_jobs:  # attach the background files to their models' rows and download lists
        with clock("background_wait"):
            written = fut.result()
        for path in written:
            rel = str(path.relative_to(out_dir))
            rows[k]["files"].append(rel)
            scene_models[k]["files"].append({"label": path.name, "path": rel})
    pool.shutdown()
    if viewer and scene_models:
        for item in ("index.html", "explorer.js", "vendor"):
            src, dst = VIEWER / item, out_dir / item
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        gen = out_dir / "generated" / "result.json"  # a generated object placed earlier (drone3d.generate) stays linked
        if gen.is_file():
            from drone3d.generate import NOTE

            try:
                g = json.loads(gen.read_text())
                aligned = g.get("aligned") or {}
                k = int(aligned.get("model", -1))
                if g.get("status") == "ok" and aligned.get("placed") and 0 <= k < len(scene_models):
                    scene_models[k]["generated"] = {"mesh": "generated/object_aligned.glb", "note": NOTE}
            except (ValueError, TypeError):
                pass
        (out_dir / "scene.json").write_text(json.dumps({"title": title, "models": scene_models}, indent=1))
    return {"models": rows, "viewer": str(out_dir / "index.html") if viewer and scene_models else None,
            "timing_s": clock.rounded()}  # fmt: skip

