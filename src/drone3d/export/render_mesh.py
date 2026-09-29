"""Offline renders of a textured mesh by ray casting: stills and fly-throughs, no OpenGL.

Headless machines rarely have a working GL stack; Open3D's ray casting (Embree,
CPU) does. Each pixel's ray gives the hit triangle and barycentric position,
which index the baked atlas (``texture_gpu``): the image is the texture as the
viewer shows it, unlit. A fly-through follows the keyframe cameras smoothly
(Catmull-Rom + slerp, ``splat.render.camera_path``) and is encoded with NVENC
when available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

__all__ = ["MeshRenderer", "load_parts", "render_mesh_flythrough", "render_parts"]


class MeshRenderer:
    """Ray-cast renderer for a triangle-soup-textured mesh (corner UVs, OBJ convention)."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, albedo: np.ndarray,
                 background: tuple[int, int, int] = (11, 29, 51)) -> None:  # fmt: skip
        import open3d as o3d
        import open3d.core as o3c

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3c.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
                                 o3c.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)))  # fmt: skip
        self.uv = np.asarray(uv, dtype=np.float32).reshape(-1, 3, 2)
        self.albedo = np.asarray(albedo, dtype=np.uint8)
        self.bg = np.array(background, dtype=np.uint8)
        self._o3c = o3c

    def render(self, f: float, cx: float, cy: float, rotation: np.ndarray, centre: np.ndarray,
               size: tuple[int, int]) -> np.ndarray:  # fmt: skip
        """Pinhole ``f, cx, cy`` (COLMAP convention), cam_from_world ``rotation``, camera ``centre`` -> RGB."""
        w, h = size
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.stack([(xs + 0.5 - cx) / f, (ys + 0.5 - cy) / f, np.ones_like(xs)], -1).reshape(-1, 3)
        d = d @ np.asarray(rotation, dtype=np.float32)  # R^T d
        o = np.broadcast_to(np.asarray(centre, dtype=np.float32), d.shape)
        rays = np.ascontiguousarray(np.concatenate([o, d], 1), dtype=np.float32)  # numpy-scalar params promote to float64
        hit = self.scene.cast_rays(self._o3c.Tensor(rays))
        prim = hit["primitive_ids"].numpy().astype(np.int64)
        bary = hit["primitive_uvs"].numpy()
        ok = np.isfinite(hit["t_hit"].numpy()) & (prim >= 0) & (prim < len(self.uv))
        out = np.tile(self.bg, (h * w, 1))
        if ok.any():
            p, b = prim[ok], bary[ok]
            tuv = (1 - b[:, :1] - b[:, 1:]) * self.uv[p, 0] + b[:, :1] * self.uv[p, 1] + b[:, 1:] * self.uv[p, 2]
            s = self.albedo.shape[0]
            tx = np.clip(tuv[:, 0] * s - 0.5, 0, s - 1.001)
            ty = np.clip((1.0 - tuv[:, 1]) * s - 0.5, 0, s - 1.001)
            x0, y0 = tx.astype(np.int64), ty.astype(np.int64)
            fx, fy = (tx - x0)[:, None], (ty - y0)[:, None]
            a = self.albedo
            col = (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
                   + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)  # fmt: skip
            out[ok] = col.round().clip(0, 255).astype(np.uint8)
        return out.reshape(h, w, 3)


def load_parts(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]]:
    """A mesh file's parts in the z-up export frame -> [(vertices, faces, corner UVs [F, 3, 2] (v up) or None,
    albedo or None)] -- for a vertex-coloured part, (vertices, faces, vertex colours, None); a GLB's node
    transforms applied."""
    import trimesh

    scene = trimesh.load(str(path), process=False, maintain_order=True)
    if isinstance(scene, trimesh.Scene):
        meshes = [scene.geometry[g].copy().apply_transform(scene.graph[n][0])
                  for n, g in ((n, scene.graph[n][1]) for n in scene.graph.nodes_geometry)]  # fmt: skip
    else:
        meshes = [scene]
    parts = []
    for m in meshes:
        v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
        if path.suffix.lower() in (".glb", ".gltf"):  # glTF is y-up: back to the export's z-up frame
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
        vis = getattr(m, "visual", None)
        mat = getattr(vis, "material", None)
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        uv = getattr(vis, "uv", None)
        ok = uv is not None and img is not None
        if not ok and getattr(vis, "kind", None) == "vertex":  # vertex colours: albedo None, the colours in place of UV
            parts.append((v, f, np.asarray(vis.vertex_colors)[:, :3].astype(np.float64), None))
            continue
        parts.append((v, f, np.asarray(uv)[f] if ok else None, np.asarray(img.convert("RGB")) if ok else None))
    return parts


def render_parts(parts, fpx: float, R: np.ndarray, c: np.ndarray, size: tuple[int, int], *,  # type: ignore[no-untyped-def]
                 background: tuple[int, int, int] = (11, 29, 51)) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast ``load_parts`` parts together from a pinhole camera (``fpx``, principal point at the centre,
    cam_from_world ``R``, centre ``c``), unlit: the nearest hit wins -> (RGB [h, w, 3], hit mask [h, w])."""
    import open3d as o3d
    import open3d.core as o3c

    w, h = size
    scene = o3d.t.geometry.RaycastingScene()
    ids = [scene.add_triangles(o3c.Tensor(v.astype(np.float32)), o3c.Tensor(f.astype(np.uint32))) for v, f, _, _ in parts]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (np.stack([(xs + 0.5 - w / 2) / fpx, (ys + 0.5 - h / 2) / fpx, np.ones_like(xs)], -1).reshape(-1, 3) @ R).astype(np.float32)
    hit = scene.cast_rays(o3c.Tensor(np.concatenate([np.broadcast_to(c, d.shape).astype(np.float32), d], 1)))
    geo, prim, bary = hit["geometry_ids"].numpy(), hit["primitive_ids"].numpy().astype(np.int64), hit["primitive_uvs"].numpy()
    ok = np.isfinite(hit["t_hit"].numpy())
    out = np.tile(np.array(background, np.uint8), (h * w, 1))
    for gid, (_, _, uv, albedo) in zip(ids, parts, strict=True):
        sel = ok & (geo == gid)
        if not sel.any():
            continue
        p, b = prim[sel], bary[sel]
        if uv is None:
            out[sel] = (200, 200, 200)
            continue
        if albedo is None:  # vertex colours
            ff = parts[ids.index(gid)][1][p]
            out[sel] = ((1 - b[:, :1] - b[:, 1:]) * uv[ff[:, 0]] + b[:, :1] * uv[ff[:, 1]] + b[:, 1:] * uv[ff[:, 2]]).astype(np.uint8)
            continue
        tuv = (1 - b[:, :1] - b[:, 1:]) * uv[p, 0] + b[:, :1] * uv[p, 1] + b[:, 1:] * uv[p, 2]
        sh, sw = albedo.shape[:2]
        tx = np.clip(tuv[:, 0] * sw - 0.5, 0, sw - 1).astype(np.int64)
        ty = np.clip((1.0 - tuv[:, 1]) * sh - 0.5, 0, sh - 1).astype(np.int64)
        out[sel] = albedo[ty, tx]
    return out.reshape(h, w, 3), ok.reshape(h, w)


def _encode(frames_dir: Path, fps: int, out: Path) -> None:
    from drone3d.io.nvdec import ffmpeg_bin

    ff = ffmpeg_bin()
    for codec in (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"], ["-c:v", "libx264", "-crf", "18"]):
        res = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                              "-i", str(frames_dir / "%05d.jpg"), *codec, "-pix_fmt", "yuv420p", str(out)],
                             capture_output=True, check=False)  # fmt: skip
        if res.returncode == 0:
            return
    raise RuntimeError(f"ffmpeg could not encode {out}: {res.stderr.decode(errors='replace')[-300:]}")


def render_mesh_flythrough(model_export_dir: Path, sfm_model: Path, out: Path, *, seconds: float = 10.0,
                           fps: int = 30, size: tuple[int, int] = (1920, 1080), pullback: float = 0.15) -> Path:  # fmt: skip
    """Fly-through of ``model_export_dir/mesh_textured.obj`` along the keyframe cameras of ``sfm_model``.

    ``frame.json`` (written by the export stage) maps the SfM frame to the export
    frame; cameras are moved with it. ``pullback`` backs each camera away from
    the scene by that share of the median scene depth, to show more context.
    """
    import cv2
    import pycolmap
    import trimesh

    from drone3d.splat.render import camera_path

    tm = trimesh.load(model_export_dir / "mesh_textured.obj", force="mesh", process=False)
    faces = np.asarray(tm.faces)
    uv = np.asarray(tm.visual.uv)[faces]  # corner UVs
    albedo = np.asarray(tm.visual.material.image.convert("RGB"))
    renderer = MeshRenderer(np.asarray(tm.vertices), faces, uv, albedo)
    frame = json.loads((model_export_dir / "frame.json").read_text())
    s, r, t = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])
    rec = pycolmap.Reconstruction(str(sfm_model))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    cam = rec.cameras[ims[0].camera_id]
    centres = np.array([s * r @ im.projection_center() + t for im in ims])
    rots = np.array([im.cam_from_world().rotation.matrix() @ r.T for im in ims])
    n = max(2, int(seconds * fps))
    path_c, path_r = camera_path(centres, rots, n)
    k = size[0] / cam.width
    f, cx, cy = cam.params[0] * k, cam.params[1] * k, cam.params[2] * k + (size[1] - cam.height * k) / 2  # same scale, centred
    depth = float(np.median(np.linalg.norm(np.asarray(tm.vertices) - centres.mean(0), axis=1)))
    tmp = out.parent / (out.stem + "_frames")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for i, (c, rr) in enumerate(zip(path_c, path_r, strict=True)):
        eye = c - pullback * depth * rr[2]  # back along the optical axis
        img = renderer.render(f, cx, cy, rr, eye, size)
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    _encode(tmp, fps, out)
    shutil.rmtree(tmp)
    return out
