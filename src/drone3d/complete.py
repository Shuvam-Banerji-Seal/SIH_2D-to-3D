"""The complete model of a run's subject: the whole object, textured on every side, in its measured scene.

A flight that circles a subject part of the way measures part of it: the highrise orbit flies 110 degrees
of heading, so its fused model is two walls and a corner -- seen from the far side, a shell one looks
into, and splats that smear. ``drone3d generate`` places TRELLIS.2's whole object on the measurement
(drone3d.generate: upright ICP, 89 % of the measured tower within 4 % of the subject radius of it). This
module makes that object the model:

- **texture** -- every texel of the generated object's own UV atlas is put back on its surface and coloured
  from the keyframes that see it: a depth map of the whole scene per keyframe for occlusion, the export
  baker's exposure gains and lens model, the two best views blended. So the walls the flight filmed carry
  the photographs; texels no keyframe sees keep the generated colour, moved to the photographs' statistics
  (where both exist) and blended into them by viewing angle;
- **no platform** -- a slab the generator stood the object on goes, outside what stands on it (Jal Mahal's
  lake);
- **planes** -- the object's large planes (facades, roofs) are made flat: the generator's window recesses are
  not the real facade's, and photographs projected onto them smear;
- **carving** -- faces the keyframes saw through (nearer than the measured surface behind them) go: what the
  generator invented where the flight looked is replaced by what it saw;
- **scene** -- the measured shell the object now stands in (inside it, or within 2 % of its size) gives way;
  the terrain and whatever else stands beside it stay;
- **ground** -- where the flight never saw the ground round the subject, a grid at the terrain's height with
  inpainted colours fills it (a separate node, ``ground_fill``);
- **solid** -- a watertight copy for STL (voxelised with its base capped, marching cubes, smoothed): a print or
  a CAD import needs a closed surface.

The subject is the closed solid, in every format, textured on a triangle-soup atlas: the photographs where a
keyframe sees it, the generated texture (from the nearest point of the generated object) elsewhere. Written to ``export/complete/``: ``subject.{glb,obj,fbx,stl}`` -- upright, glTF y-up, the
pivot at the centre of its base, in the export's units -- ``scene.glb`` (the subject in the measured scene,
two named nodes) and ``result.json``. What the keyframes did not see is generated, not measured, and the result says how much.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["carve", "complete_model", "drop_base_slab", "link_scene", "retexture", "snap_planes", "solidify", "splats_360", "transfer_texture"]

log = get_logger(__name__)

Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # z-up -> glTF y-up, as export.formats
SEEN_FACING = 0.15  # |cos| of the viewing angle below which a keyframe does not texture a texel (> 81 degrees)
FULL_FACING = 0.4  # ... and above which its colour replaces the generated one entirely (< 66 degrees)
NOTE = "complete model: measured and photographed where the flight saw it; the rest generated (TRELLIS.2) and inpainted, not measured"
TEXTURE_SIZE = 4096  # the subject's atlas: TRELLIS.2 bakes 2048^2, too coarse for the photographed walls' windows


def _load_glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """A GLB's (single) mesh in the z-up frame -> vertices, faces, per-vertex UV (v up) or None, texture or None."""
    import trimesh

    scene = trimesh.load(str(path), process=False, maintain_order=True)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([scene.geometry[g].copy().apply_transform(scene.graph[n][0])
                                         for n, g in ((n, scene.graph[n][1]) for n in scene.graph.nodes_geometry)])  # fmt: skip
    else:
        mesh = scene
    v = np.asarray(mesh.vertices, np.float64) @ Y_UP  # y-up -> z-up (Y_UP is orthogonal: its inverse is its transpose)
    uv, img = None, None
    vis = getattr(mesh, "visual", None)
    if getattr(vis, "uv", None) is not None:
        mat = vis.material
        tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if tex is not None:
            uv, img = np.asarray(vis.uv, np.float64), np.asarray(tex.convert("RGB"))
    return v, np.asarray(mesh.faces, np.int64), uv, img


def _uv_texels(uv: np.ndarray, f: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The atlas texels each triangle covers -> (flat texel index, triangle, barycentric [N, 3]).

    Ray casting in UV space (Embree): a ray per texel centre, down onto the triangles laid out at (u, 1 - v).
    """
    import open3d as o3d
    import open3d.core as o3c

    tri = np.stack([uv[:, 0] * size, (1.0 - uv[:, 1]) * size, np.zeros(len(uv))], 1).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(tri), o3c.Tensor(f.astype(np.uint32)))
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    o = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5, np.ones(size * size, np.float32)], 1)
    d = np.broadcast_to(np.array([0, 0, -1], np.float32), o.shape)
    hit = scene.cast_rays(o3c.Tensor(np.ascontiguousarray(np.concatenate([o, d], 1))))
    prim = hit["primitive_ids"].numpy().astype(np.int64)
    ok = np.isfinite(hit["t_hit"].numpy()) & (prim >= 0) & (prim < len(f))
    b = hit["primitive_uvs"].numpy()[ok]
    return np.flatnonzero(ok), prim[ok], np.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)


def _depth_maps(v: np.ndarray, f: np.ndarray, views, width: int = 640) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    """Per keyframe, the camera-frame depth of the nearest surface of (v, f) -> [h, w] maps at ``width`` px."""
    import open3d as o3d
    import open3d.core as o3c
    import torch

    from drone3d.fastsfm.dense import _undistort

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(v.astype(np.float32)), o3c.Tensor(f.astype(np.uint32)))
    out = []
    for vw in views:
        h0, w0 = vw.image.shape[:2]
        s = width / w0
        w, h = width, max(1, round(h0 * s))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        # pixel centres of this map in the view's own pixels (texture_gpu._project's convention: -0.5)
        px, py = (xs + 0.5) / s, (ys + 0.5) / s
        xn, yn = _undistort(torch.as_tensor((px - vw.cx) / vw.fx), torch.as_tensor((py - vw.cy) / vw.fx), vw.k1)
        dc = np.stack([xn.numpy(), yn.numpy(), np.ones_like(xs)], -1).reshape(-1, 3)  # camera z = 1
        r = np.asarray(vw.rotation)
        centre = -r.T @ np.asarray(vw.translation)
        rays = np.concatenate([np.broadcast_to(centre, dc.shape), dc @ r], 1).astype(np.float32)
        t = scene.cast_rays(o3c.Tensor(rays))["t_hit"].numpy().reshape(h, w)  # along an unnormalised ray: t = z
        out.append(np.where(np.isfinite(t), t, np.inf).astype(np.float32))
    return out


def _sky_masks(views, width: int = 640, model: str = "depth-anything/Depth-Anything-V2-Large-hf") -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    """Per keyframe, the sky at ``width`` px, by the dense stage's rule (Depth Anything V2: disparity under 0.5 %
    of the image's maximum), eroded 3 px so that a silhouette is never sky."""
    import torch
    import torch.nn.functional as F
    from scipy import ndimage

    from drone3d.fastsfm.mono import MonoDepth

    mono = MonoDepth(model)
    out = []
    for vw in views:
        h0, w0 = vw.image.shape[:2]
        h = max(1, round(h0 * width / w0))
        small = F.interpolate(vw.image.permute(2, 0, 1)[None].float(), size=(h, width), mode="area")[0].permute(1, 2, 0)
        disp = mono(small.round().clamp(0, 255).to(torch.uint8)[None])[0]
        sky = (disp <= mono.sky_rel * disp.max().clamp_min(1e-6)).cpu().numpy()
        out.append(ndimage.binary_erosion(sky, iterations=3))
    del mono
    torch.cuda.empty_cache()
    return out


def drop_base_slab(v: np.ndarray, f: np.ndarray, *, band: float = 0.02, min_share: float = 0.15) -> np.ndarray:
    """Faces of the platform a generator stood its object on -> keep mask.

    TRELLIS.2 gave Jal Mahal its lake: a horizontal slab at a third of the palace's height, 52 % of the
    object's area, the reflection painted on it. The platform is the ``band``-thick height band in the object's
    lower half with the most horizontal area, when that is ``min_share`` of the area or more; its faces
    outside the footprint of what stands on it go, and everything below it (under ground or water). The
    highrise's largest horizontal areas are its roofs (6 % each, at the top): nothing goes.
    """
    from scipy import ndimage

    z0, height = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    tri = v[f]
    cen = tri.mean(1)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    flat = np.abs(cr[:, 2]) / np.maximum(2 * area, 1e-12) > 0.9
    rel = (cen[:, 2] - z0) / max(height, 1e-12)
    bins = np.floor(rel / band).astype(int)
    lower = flat & (rel < 0.5)
    if not lower.any():
        return np.ones(len(f), bool)
    per = np.bincount(bins[lower], weights=area[lower])
    k = int(per.argmax())
    if per[k] < min_share * area.sum():
        return np.ones(len(f), bool)
    zp = z0 + (k + 0.5) * band * height
    cell = float(np.ptp(v[:, :2], 0).max()) / 200
    lo = v[:, :2].min(0) - 4 * cell
    shape = tuple((np.ptp(v[:, :2], 0) / cell).astype(int)[::-1] + 9)
    ij = np.floor((tri[cen[:, 2] > zp + 2.5 * band * height].reshape(-1, 3)[:, :2] - lo) / cell).astype(int)
    foot = np.zeros(shape, bool)
    foot[ij[:, 1], ij[:, 0]] = True
    foot = ndimage.binary_dilation(ndimage.binary_fill_holes(ndimage.binary_closing(foot, iterations=2)), iterations=3)
    cj = np.floor((cen[:, :2] - lo) / cell).astype(int)
    out = ~foot[cj[:, 1], cj[:, 0]]
    below = cen[:, 2] < zp - band * height  # under the platform (the ground, the water) is not the object
    return ~((out & (cen[:, 2] < zp + 2.5 * band * height)) | below)  # a thick slab: its upper face too


def snap_planes(v: np.ndarray, f: np.ndarray, *, tol: float = 0.025, min_share: float = 0.04, max_planes: int = 16,
                seed: int = 0) -> tuple[np.ndarray, list[dict]]:  # fmt: skip
    """The object's large planes made flat -> ``(vertices, planes)``.

    A generator models a facade's windows as recesses 1-2 % of the building deep, where the real facade is flat
    to the camera: photographs projected onto relief that is not there smear with the parallax (the highrise's
    window columns ran). Planes holding ``min_share`` of the area (RANSAC, area-weighted, then a weighted PCA
    fit) take every vertex within ``tol`` of the object's size of them, inside their extent; a vertex near two
    goes to their intersection. A statue or a dome has no such plane and is left as it is.
    """
    rng = np.random.default_rng(seed)
    size = float(np.ptp(v, 0).max())
    _, weld = np.unique(np.round(v / (1e-6 * size)), axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    wv = np.zeros((weld.max() + 1, 3))
    wv[weld] = v
    wf = weld[f]
    tri = wv[wf]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(2 * area, 1e-12)[:, None]
    cen = tri.mean(1)
    free = area > 0
    planes = []
    for _ in range(max_planes):
        idx = np.flatnonzero(free)
        if not len(idx) or area[idx].sum() < min_share * area.sum():
            break
        p = area[idx] / area[idx].sum()
        sample = rng.choice(idx, size=min(len(idx), 20000), p=p)  # candidates scored on an area-weighted sample
        cand = rng.choice(idx, size=300, p=p)
        hit = (np.abs(((cen[sample][None] - cen[cand][:, None]) * n[cand][:, None]).sum(-1)) < tol * size) & \
              (np.abs(n[sample] @ n[cand].T).T > 0.9)  # fmt: skip
        k = cand[int(hit.sum(1).argmax())]
        inl = (np.abs((cen[idx] - cen[k]) @ n[k]) < tol * size) & (np.abs(n[idx] @ n[k]) > 0.9)
        sc, sup = area[idx][inl].sum(), idx[inl]
        if sc < min_share * area.sum():
            break
        w = area[sup] / area[sup].sum()
        c = (w[:, None] * cen[sup]).sum(0)
        cov = ((cen[sup] - c) * w[:, None]).T @ (cen[sup] - c)
        normal = np.linalg.eigh(cov)[1][:, 0]
        e1 = np.linalg.eigh(cov)[1][:, 2]
        e2 = np.cross(normal, e1)
        uv = np.stack([(cen[sup] - c) @ e1, (cen[sup] - c) @ e2], 1)
        planes.append({"normal": normal, "centre": c, "e1": e1, "e2": e2, "lo": uv.min(0) - tol * size,
                       "hi": uv.max(0) + tol * size, "share": float(sc / area.sum())})  # fmt: skip
        free[sup] = False
        free &= ~((np.abs((cen - c) @ normal) < tol * size) & (np.abs(n @ normal) > 0.5))
    if not planes:
        return v, []
    # each welded vertex: the planes it is within tol of (inside their extent), then projected
    hits, dists = [], []
    for pl in planes:
        d = (wv - pl["centre"]) @ pl["normal"]
        q = np.stack([(wv - pl["centre"]) @ pl["e1"], (wv - pl["centre"]) @ pl["e2"]], 1)
        hits.append((np.abs(d) < tol * size) & (q >= pl["lo"]).all(1) & (q <= pl["hi"]).all(1))
        dists.append(np.abs(d))
    hits, dists = np.stack(hits, 1), np.stack(dists, 1)
    # near-parallel planes (two roof levels 0.5 degrees apart) meet far away: a vertex takes the nearer one only
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if abs(planes[i]["normal"] @ planes[j]["normal"]) > 0.9:
                both = hits[:, i] & hits[:, j]
                far_i = both & (dists[:, i] > dists[:, j])
                hits[far_i, i] = False
                hits[both & ~far_i, j] = False
    out = wv.copy()
    combos, which = np.unique(hits, axis=0, return_inverse=True)
    for c, combo in enumerate(combos):  # vertices near the same planes: one projection for all of them
        if not combo.any():
            continue
        ks = np.flatnonzero(combo)[:3]  # a corner is where three planes meet
        a = np.array([planes[k]["normal"] for k in ks])
        b = np.array([planes[k]["normal"] @ planes[k]["centre"] for k in ks])
        sel = which.reshape(-1) == c
        # the point nearest each vertex on all its planes (the plane, their edge or their corner): min |x - p| s.t. A x = b
        lam = np.linalg.lstsq(a @ a.T, (wv[sel] @ a.T - b).T, rcond=None)[0]
        moved = wv[sel] - lam.T @ a
        ok = np.linalg.norm(moved - wv[sel], axis=1) <= 2 * tol * size  # never far: a bad corner stays as it was
        out[np.flatnonzero(sel)[ok]] = moved[ok]
    info = [{"normal": pl["normal"].round(3).tolist(), "share": round(pl["share"], 3)} for pl in planes]
    return out[weld], info


def carve(v: np.ndarray, f: np.ndarray, views, measured: tuple[np.ndarray, np.ndarray], *,  # type: ignore[no-untyped-def]
          sky: list[np.ndarray] | None = None, margin: float = 0.03, min_views: int = 2, window: int = 7,
          sky_margin: int = 5, device: str = "cuda") -> np.ndarray:  # fmt: skip
    """Faces of a generated object that the keyframes saw through -> keep mask.

    A face whose centroid lies nearer a keyframe than the measured surface the keyframe saw at that pixel, by
    more than ``margin`` of the depth, in ``min_views`` keyframes, stands in observed free space: the generator
    invented it (TRELLIS.2 put a canopy in front of the highrise's facade). The measured depth is min-filtered
    over ``window`` pixels at 640 px, so the object's own silhouette -- 1 % off the measured one -- is not eaten.
    ``sky`` (per keyframe, at the depth maps' size): the sky is seen through as well -- the generated crown stood
    above the real roof line -- ``sky_margin`` pixels in from its edge: Jal Mahal's generated domes, a few
    pixels off the real ones, were carved at the edge. What no keyframe looked at is never carved.
    """
    import torch
    import torch.nn.functional as F

    from drone3d.export.texture_gpu import _project

    depth = _depth_maps(measured[0], measured[1], views)
    cen = torch.as_tensor(v[f].mean(1), dtype=torch.float32, device=device)
    votes = torch.zeros(len(f), dtype=torch.int32, device=device)
    for k, vw in enumerate(views):
        h, w = vw.image.shape[:2]
        u, y, z = _project(cen, vw)
        dm = torch.as_tensor(depth[k], device=device)
        dm = -F.max_pool2d(-dm[None, None], window, stride=1, padding=window // 2)[0, 0]  # nearest surface around
        s = dm.shape[1] / w
        inside = (z > 0) & (u >= 0) & (u <= w - 1) & (y >= 0) & (y <= h - 1)
        near = dm[((y + 0.5) * s).long().clamp(0, dm.shape[0] - 1), ((u + 0.5) * s).long().clamp(0, dm.shape[1] - 1)]
        free = torch.isfinite(near) & (z < near * (1 - margin))
        if sky is not None:
            from scipy import ndimage

            sk = torch.as_tensor(ndimage.binary_erosion(sky[k], iterations=sky_margin), device=device)
            free |= sk[((y + 0.5) * s).long().clamp(0, sk.shape[0] - 1), ((u + 0.5) * s).long().clamp(0, sk.shape[1] - 1)]
        votes += (inside & free).int()
    return (votes < min_views).cpu().numpy()


def retexture(v: np.ndarray, f: np.ndarray, uv: np.ndarray | None, albedo: np.ndarray, views, *,  # type: ignore[no-untyped-def]
              occluders: tuple[np.ndarray, np.ndarray] | None = None, sky: list[np.ndarray] | None = None,
              texels: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
              device: str = "cuda") -> tuple[np.ndarray, dict]:  # fmt: skip
    """The keyframes' colours in a mesh's own UV atlas -> ``(albedo, info)``.

    ``v`` in the views' (SfM) frame; ``uv`` per vertex, v up -- or ``texels`` given directly (flat atlas index,
    triangle, barycentric: a triangle-soup layout's); ``occluders``: the rest of the scene, which may hide
    the mesh from a keyframe. Texels no keyframe sees at under 81 degrees keep ``albedo``'s colour, moved to the
    photographs' colour statistics.
    """
    import torch
    import torch.nn.functional as F
    from scipy import ndimage

    from drone3d.export.texture_gpu import _project, _view_gains

    t0 = time.perf_counter()
    size = albedo.shape[0]
    flat, tid, bary = texels if texels is not None else _uv_texels(uv, f, size)
    dev = torch.device(device)
    vt = torch.as_tensor(v, dtype=torch.float32, device=dev)
    ft = torch.as_tensor(f, dtype=torch.long, device=dev)
    tri = vt[ft]
    nrm = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tid_t = torch.as_tensor(tid, device=dev)
    pts = (torch.as_tensor(bary, dtype=torch.float32, device=dev)[:, :, None] * tri[tid_t]).sum(1)
    n_pts = nrm[tid_t]
    occ_v, occ_f = (np.concatenate([v, occluders[0]]), np.concatenate([f, occluders[1] + len(v)])) if occluders else (v, f)
    depth = _depth_maps(occ_v, occ_f, views)
    timing = {"raster": time.perf_counter() - t0}
    t0 = time.perf_counter()

    def score(p: torch.Tensor, n: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        vw = views[k]
        h, w = vw.image.shape[:2]
        u, y, z = _project(p, vw)
        inside = (z > 0) & (u >= 0) & (u <= w - 1) & (y >= 0) & (y <= h - 1)
        if vw.mask is not None:
            mh, mw = vw.mask.shape
            inside &= ~vw.mask[(y * (mh / h)).long().clamp(0, mh - 1), (u * (mw / w)).long().clamp(0, mw - 1)]
        dm = torch.as_tensor(depth[k], device=dev)
        s = dm.shape[1] / w
        near = dm[((y + 0.5) * s).long().clamp(0, dm.shape[0] - 1), ((u + 0.5) * s).long().clamp(0, dm.shape[1] - 1)]
        centre = -torch.as_tensor(vw.rotation.T @ vw.translation, dtype=torch.float32, device=dev)
        to_cam = centre - p
        dist = to_cam.norm(dim=-1).clamp_min(1e-9)
        facing = (n * to_cam).sum(-1).abs() / dist
        ok = inside & (z <= near * 1.02) & (facing >= SEEN_FACING)
        if sky is not None:  # a texel that lands on the sky is not seen, whatever the geometry says
            sk = torch.as_tensor(sky[k], device=dev)
            ok &= ~sk[((y + 0.5) * s).long().clamp(0, sk.shape[0] - 1), ((u + 0.5) * s).long().clamp(0, sk.shape[1] - 1)]
        return torch.where(ok, facing / dist, torch.zeros_like(dist)), torch.where(ok, facing, torch.zeros_like(dist))

    # per triangle (centroids) for the exposure gains, per texel for the colours: the two best views
    cen = tri.mean(1)
    tri_scores = torch.stack([score(cen, nrm, k)[0] for k in range(len(views))])
    gains, gain_stats = _view_gains(views, cen, torch.where(tri_scores > 0, tri_scores, -torch.ones_like(tri_scores)))
    best = torch.zeros(2, len(pts), device=dev)
    best_k = torch.full((2, len(pts)), -1, dtype=torch.long, device=dev)
    face = torch.zeros(len(pts), device=dev)
    for k in range(len(views)):
        s, fc = score(pts, n_pts, k)
        better = s > best[0]
        second = ~better & (s > best[1])
        best[1] = torch.where(better, best[0], torch.where(second, s, best[1]))
        best_k[1] = torch.where(better, best_k[0], torch.where(second, torch.full_like(best_k[1], k), best_k[1]))
        best[0] = torch.where(better, s, best[0])
        best_k[0] = torch.where(better, torch.full_like(best_k[0], k), best_k[0])
        face = torch.where(better, fc, face)
    # the best view nearly alone (weights ~ score^4): two views 1 % apart in geometry doubled the windows
    wts = torch.where((best > 0) & (best >= 0.8 * best[:1]), best.pow(4), torch.zeros_like(best))
    wts = wts / wts.sum(0, keepdim=True).clamp_min(1e-12)
    photo = torch.zeros(len(pts), 3, device=dev)
    for rank in range(2):
        for k, vw in enumerate(views):
            sel = (best_k[rank] == k) & (wts[rank] > 0)
            if not bool(sel.any()):
                continue
            h, w = vw.image.shape[:2]
            u, y, _ = _project(pts[sel], vw)
            g = torch.stack([2 * u / (w - 1) - 1, 2 * y / (h - 1) - 1], -1)[None, None]
            sample = F.grid_sample(vw.image.permute(2, 0, 1)[None].float(), g, mode="bilinear", align_corners=True,
                                   padding_mode="border")[0, :, 0].T  # fmt: skip
            photo[sel] += wts[rank, sel, None] * (sample * gains[k] if gains is not None else sample)
    timing["views"] = time.perf_counter() - t0
    alpha = ((face - SEEN_FACING) / (FULL_FACING - SEEN_FACING)).clamp(0, 1)
    alpha = torch.where(best[0] > 0, alpha, torch.zeros_like(alpha))
    gen = torch.as_tensor(albedo.reshape(-1, 3)[flat], dtype=torch.float32, device=dev)
    # the generated colours moved onto the photographs': per channel, the mean and spread where the photographs
    # are sure (Reinhard's colour transfer; an affine 3x3 map was ill-conditioned on near-grey facades)
    sure = alpha >= 0.99
    colour_map = None
    if int(sure.sum()) >= 1000:
        gm, gs = gen[sure].mean(0), gen[sure].std(0).clamp_min(1.0)
        pm, ps = photo[sure].mean(0), photo[sure].std(0).clamp_min(1.0)
        gain = (ps / gs).clamp(0.5, 2.0)
        colour_map = torch.stack([gain, pm - gain * gm])  # [2, 3]: gain, offset
        gen = gen * colour_map[0] + colour_map[1]
    col = alpha[:, None] * photo + (1 - alpha[:, None]) * gen
    out = albedo.reshape(-1, 3).copy()
    out[flat] = col.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    # gutters: the few texels round each UV island take the nearest island texel, so filtering does not bleed
    covered = np.zeros(size * size, bool)
    covered[flat] = True
    covered = covered.reshape(size, size)
    dist, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    gut = (~covered) & (dist <= 4)
    out = out.reshape(size, size, 3)
    out[gut] = out[iy[gut], ix[gut]]
    seen_tri = (tri_scores > 0).any(0)
    info = {"texels": int(len(flat)), "photo_texels": round(float((alpha > 0).float().mean()), 4),
            "photo_texels_full": round(float(sure.float().mean()), 4),
            "seen_triangles": round(float(seen_tri.float().mean()), 4), "views": len(views),
            "colour_map": colour_map.cpu().numpy().round(4).tolist() if colour_map is not None else None,
            "gain": gain_stats, "timing_s": {k: round(x, 2) for k, x in timing.items()}}  # fmt: skip
    del pts, tri_scores
    torch.cuda.empty_cache()
    return out, info


def solidify(v: np.ndarray, f: np.ndarray, *, resolution: int = 224, yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """A closed surface of the solid (v, f) bounds (z up) -> ``(vertices, faces, info)``.

    Voxelised at ``resolution`` along its longest side in its own heading (``yaw``: its largest wall's normal,
    so walls fall on voxel planes and come out flat), its base capped, marching cubes, outward normals, bodies
    under 1 % dropped, lightly Taubin-smoothed (vertices only: the topology stays closed). Inside is what a
    flood fill from outside does not reach; where a hole let the fill in -- carving opened the highrise's
    crown, and the 3D fill added nothing to the shell -- inside is what is enclosed along two of the three axes.
    """
    import open3d as o3d
    import trimesh
    from scipy import ndimage
    from skimage.measure import marching_cubes

    c, sn = np.cos(-yaw), np.sin(-yaw)
    rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    vr = v @ rz.T
    lo, hi = vr.min(0), vr.max(0)
    vox = float((hi - lo).max()) / resolution
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vr), o3d.utility.Vector3iVector(f))
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, vox, lo - 3 * vox, hi + 3 * vox)
    idx = np.array([x.grid_index for x in grid.get_voxels()])
    shape = np.ceil((hi - lo + 6 * vox) / vox).astype(int) + 1
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    base = int(idx[:, 2].min())
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(occ.any(2), iterations=2))
    occ[:, :, base : base + 2] |= foot[:, :, None]  # the base, which a placed object is trimmed open at
    shell = ndimage.binary_dilation(occ, iterations=1)
    solid = ndimage.binary_erosion(ndimage.binary_fill_holes(shell), iterations=1) | occ
    rule = "flood fill"
    if solid.sum() - occ.sum() < 0.5 * occ.sum():  # hollow: the fill came in through a hole
        enclosed = sum((np.maximum.accumulate(occ, axis=a) & np.flip(np.maximum.accumulate(np.flip(occ, a), axis=a), a))
                       .astype(np.int8) for a in range(3))  # fmt: skip
        solid = ndimage.binary_fill_holes(occ | (enclosed >= 2))
        rule = "enclosed on two axes"
    verts, faces, _, _ = marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
    m = trimesh.Trimesh((verts - 1) * vox + lo - 3 * vox, faces, process=True)
    parts = m.split(only_watertight=False)
    m = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.01 * len(m.faces)]) if len(parts) > 1 else m
    if m.volume < 0:
        m.invert()
    trimesh.smoothing.filter_taubin(m, iterations=5)
    info = {"voxel": round(vox, 5), "fill": rule, "watertight": bool(m.is_watertight), "volume": round(float(m.volume), 4)}
    return np.asarray(m.vertices, np.float64) @ rz, np.asarray(m.faces, np.int64), info


def transfer_texture(v: np.ndarray, f: np.ndarray, src: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], *,
                     size: int = 4096, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, tuple]:  # fmt: skip
    """A triangle-soup atlas for (v, f) coloured from the nearest point of a textured mesh ``src`` = (vertices,
    faces, per-vertex UV (v up), albedo) -> ``(corner UV [F, 3, 2], albedo, texels)`` (texels as ``retexture``
    takes them)."""
    import open3d as o3d
    import open3d.core as o3c
    import torch
    import torch.nn.functional as F

    from drone3d.export.texture_gpu import soup_layout, soup_texels

    sv, sf, suv, salb = src
    dev = torch.device(device)
    tri = torch.as_tensor(v, dtype=torch.float32, device=dev)[torch.as_tensor(f, device=dev)]
    corners, uv_obj = soup_layout(len(f), size, dev)
    flat, tid, ok, (wa, wb, wc), pts = soup_texels(corners, tri, size)
    texels = (flat[ok].cpu().numpy(), tid[ok].cpu().numpy(), torch.stack([wa, wb, wc], 1)[ok].cpu().numpy())
    flat, pts = texels[0], pts[ok].cpu().numpy()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3c.Tensor(sv.astype(np.float32)), o3c.Tensor(sf.astype(np.uint32)))
    albedo = np.zeros((size * size, 3), np.uint8)
    img = torch.as_tensor(salb, device=dev).permute(2, 0, 1)[None].float()
    for s in range(0, len(pts), 4_000_000):
        q = scene.compute_closest_points(o3c.Tensor(pts[s : s + 4_000_000].astype(np.float32)))
        prim = q["primitive_ids"].numpy().astype(np.int64)
        b = q["primitive_uvs"].numpy()
        tuv = (1 - b[:, :1] - b[:, 1:]) * suv[sf[prim, 0]] + b[:, :1] * suv[sf[prim, 1]] + b[:, 1:] * suv[sf[prim, 2]]
        g = torch.as_tensor(np.stack([2 * tuv[:, 0] - 1, 1 - 2 * tuv[:, 1]], 1), dtype=torch.float32, device=dev)
        col = F.grid_sample(img, g[None, None], mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T
        albedo[flat[s : s + 4_000_000]] = col.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return uv_obj.cpu().numpy(), albedo.reshape(size, size, 3), texels


def fill_ground(v: np.ndarray, f: np.ndarray, colours: np.ndarray, centre: np.ndarray, radius: float, cell: float,
                subject: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Ground where no surface is, within ``radius`` of ``centre`` (z up) -> (vertices, faces, vertex colours).

    An orbit that flies part of the way sees the ground on its own side of the subject: seen from the other
    side, the complete subject stands at the edge of a hole. The fill is a grid at the terrain's height
    (``export.terrain.terrain_surface``: the lowest surfaces, opened, inpainted), 1 % of a cell under it so
    it never covers a measured surface, coloured by inpainting the colours of the lowest surface in the cells
    around it. ``colours``: per face; ``subject``: its vertices and faces, whose footprint is not filled.
    """
    from scipy import ndimage

    from drone3d.export.terrain import _fill, terrain_surface

    x0, y0 = centre[0] - radius, centre[1] - radius
    n = int(np.ceil(2 * radius / cell))
    gv = v[np.linalg.norm(v[:, :2] - centre[:2], axis=1) < 1.5 * radius]
    if len(gv) < 100:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3), np.uint8)
    ground, _, (gx0, gy0) = terrain_surface(gv, cell, radius / 2)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    xs, ys = x0 + (jj + 0.5) * cell, y0 + (ii + 0.5) * cell
    from scipy.ndimage import map_coordinates

    z = map_coordinates(ground, [(ys - gy0) / cell - 0.5, (xs - gx0) / cell - 0.5], order=1, mode="nearest")
    # cells a surface covers (seen from above), and each covered cell's lowest surface colour
    tri = v[f]
    samples = np.concatenate([tri.mean(1), tri.reshape(-1, 3)])
    sc = np.concatenate([colours, np.repeat(colours, 3, axis=0)]).astype(np.float64)
    ij = np.floor((samples[:, :2] - (x0, y0)) / cell).astype(int)
    inb = (ij >= 0).all(1) & (ij < n).all(1)
    ij, sz, sc = ij[inb], samples[inb, 2], sc[inb]
    flat = ij[:, 1] * n + ij[:, 0]
    order = np.lexsort((sz, flat))
    first = order[np.r_[True, np.diff(flat[order]) != 0]]
    covered = np.zeros(n * n, bool)
    covered[flat[first]] = True
    col = np.zeros((n * n, 3))
    col[flat[first]] = sc[first]
    seen = covered.reshape(n, n)  # cells with a colour
    col = np.stack([_fill(col[:, c].reshape(n, n), seen, iters=200) for c in range(3)], -1)
    covered = ndimage.binary_closing(seen, iterations=1)  # pinholes between samples are not holes
    hole = ~covered & (np.hypot(xs - centre[0], ys - centre[1]) < radius)
    if subject is not None:  # the subject stands there
        sj = np.floor((subject[0][:, :2] - (x0, y0)) / cell).astype(int)
        sj = sj[(sj >= 0).all(1) & (sj < n).all(1)]
        mask = np.zeros((n, n), bool)
        mask[sj[:, 1], sj[:, 0]] = True
        hole &= ~ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    hole = ndimage.binary_dilation(hole, iterations=1)  # overlap the seen ground's ragged edge
    q = np.arange(n * n).reshape(n, n)
    ok = (hole[:-1, :-1] | hole[:-1, 1:] | hole[1:, :-1] | hole[1:, 1:]).ravel()
    a, b, c, d = q[:-1, :-1].ravel()[ok], q[:-1, 1:].ravel()[ok], q[1:, :-1].ravel()[ok], q[1:, 1:].ravel()[ok]
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    used, inv = np.unique(faces, return_inverse=True)
    verts = np.stack([xs.ravel(), ys.ravel(), z.ravel() - 0.01 * cell], 1)[used]
    return verts, inv.reshape(-1, 3), col.reshape(-1, 3)[used].round().clip(0, 255).astype(np.uint8)


def _face_colours(mesh) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Per face of a textured trimesh, the texture at its centroid (uint8 [F, 3])."""
    uv = np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)].mean(1)
    mat = mesh.visual.material
    img = np.asarray((getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)).convert("RGB"))
    h, w = img.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    return img[y, x]


def _textured(v: np.ndarray, f: np.ndarray, corner_uv: np.ndarray, image):  # type: ignore[no-untyped-def]
    """A glTF-ready (y-up) trimesh of a triangle-soup-textured mesh: glTF's attributes are per vertex, so each
    corner is its own vertex."""
    import trimesh

    return trimesh.Trimesh(vertices=v[f].reshape(-1, 3) @ Y_UP.T, faces=np.arange(3 * len(f)).reshape(-1, 3),
                           visual=trimesh.visual.TextureVisuals(uv=corner_uv.reshape(-1, 2), image=image), process=False)  # fmt: skip


def _write_subject(v: np.ndarray, f: np.ndarray, corner_uv: np.ndarray, albedo: np.ndarray, stem: Path) -> list[Path]:
    """Textured GLB (node ``subject``, JPEG texture) and OBJ (shared vertices, per-corner ``vt``; + .mtl, .jpg)."""
    import io

    import trimesh
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(albedo).save(buf, format="JPEG", quality=92)
    tex = stem.with_name(stem.name + "_albedo.jpg")
    tex.write_bytes(buf.getvalue())
    glb = stem.with_suffix(".glb")
    trimesh.Scene({"subject": _textured(v, f, corner_uv, Image.open(io.BytesIO(buf.getvalue())))}).export(glb)
    obj, mtl = stem.with_suffix(".obj"), stem.with_suffix(".mtl")
    mtl.write_text(f"newmtl albedo\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd {tex.name}\n")
    fi = np.empty((len(f), 6), np.int64)
    fi[:, 0::2] = f + 1
    fi[:, 1::2] = np.arange(3 * len(f)).reshape(-1, 3) + 1
    with obj.open("w") as fh:
        fh.write(f"mtllib {mtl.name}\nusemtl albedo\n")
        np.savetxt(fh, v, fmt="v %.5f %.5f %.5f")
        np.savetxt(fh, corner_uv.reshape(-1, 2), fmt="vt %.6f %.6f")
        np.savetxt(fh, fi, fmt="f %d/%d %d/%d %d/%d")
    return [glb, obj, mtl, tex]


def complete_model(run_dir: Path, *, model: int = 0, texture_views: int = 48) -> dict:
    """The complete subject and scene of a run whose generated object is placed (drone3d.generate)."""
    import pycolmap
    import torch
    from torchvision.io import read_file

    from drone3d.export.formats import write_fbx
    from drone3d.export.texture_gpu import View
    from drone3d.gpu.nvjpeg import decode_jpeg
    from drone3d.io.overlay import load_mask

    t_start = time.perf_counter()
    gen_dir = run_dir / "export" / "generated"
    gen = json.loads((gen_dir / "result.json").read_text()) if (gen_dir / "result.json").is_file() else {}
    aligned = gen.get("aligned") or {}
    if gen.get("status") != "ok" or not aligned.get("placed") or int(aligned.get("model", -1)) != model:
        raise ValueError("no generated object placed on this model: run `drone3d generate` first")
    sfm = json.loads((run_dir / "sfm" / "result.json").read_text())
    model_dir = Path(sfm["models"][model]["path"])
    name = model_dir.name
    mdir = run_dir / "export" / f"model_{name}"
    frame = json.loads((mdir / "frame.json").read_text())
    fs, fr, ft = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])

    def to_sfm(x: np.ndarray) -> np.ndarray:  # export frame (z up) -> SfM frame
        return ((x - ft) / fs) @ fr

    gv, gf, guv, galb = _load_glb(gen_dir / "object_aligned.glb")
    if guv is None:
        raise ValueError("the generated object has no texture")
    sv, sf, _, _ = _load_glb(mdir / "mesh_textured.glb")  # the clean scene (triangle soup)

    # the keyframe views, as the export's baker builds them
    rec = pycolmap.Reconstruction(str(model_dir))
    posed = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    pick = [posed[int(round(i))] for i in np.linspace(0, len(posed) - 1, min(texture_views, len(posed)))]
    images = run_dir / "dataset" / "images"
    overlay = load_mask(images.parent)
    overlay = torch.from_numpy(overlay).cuda() if overlay is not None else None
    views = []
    for im in pick:
        img = decode_jpeg(read_file(str(images / im.name)), device="cuda").permute(1, 2, 0).contiguous()
        cam = rec.cameras[im.camera_id]
        s = img.shape[1] / cam.width
        pose = im.cam_from_world()
        views.append(View(cam.params[0] * s, cam.params[1] * s, cam.params[2] * s, np.asarray(pose.rotation.matrix()),
                          np.asarray(pose.translation), img,
                          float(cam.params[3]) if cam.model.name in ("SIMPLE_RADIAL", "RADIAL") else 0.0, overlay))  # fmt: skip
    import open3d as o3d
    import open3d.core as o3c
    import trimesh

    # 1. no ground disc, flat facades, then carve what the keyframes saw through
    t0 = time.perf_counter()
    slab_keep = drop_base_slab(gv, gf)
    gf = gf[slab_keep]
    gv, planes = snap_planes(gv, gf)
    walls = [pl for pl in planes if abs(pl["normal"][2]) < 0.3]
    yaw = float(np.arctan2(walls[0]["normal"][1], walls[0]["normal"][0])) if walls else 0.0
    raw = o3d.io.read_triangle_mesh(str(run_dir / "dense" / f"model_{name}" / "mesh.ply"))  # the measurement
    sky = _sky_masks(views)
    keep = carve(to_sfm(gv), gf, views, (np.asarray(raw.vertices), np.asarray(raw.triangles)), sky=sky)
    carved = int((~keep).sum())
    # what the carving cut loose (pieces under 1 %), connected through positions: the GLB splits its vertices
    # along UV seams, 13108 index-connected pieces of what is one body
    _, weld = np.unique(np.round(gv / (1e-6 * float(np.ptp(gv, 0).max())), 0), axis=0, return_inverse=True)
    kf = gf[keep]
    adj = trimesh.graph.face_adjacency(weld.reshape(-1)[kf])
    comps = trimesh.graph.connected_components(adj, nodes=np.arange(len(kf)), min_len=1)
    big = [c for c in comps if len(c) >= 0.01 * len(kf)]
    gf = kf[np.sort(np.concatenate(big))] if big else kf
    used = np.unique(gf)
    remap = np.full(len(gv), -1)
    remap[used] = np.arange(len(used))
    gv, guv, gf = gv[used], guv[used], remap[gf]
    t_carve = time.perf_counter() - t0

    # 2. the solid, and with it the measured shell that gives way: measured triangles inside the subject or
    #    within 2 % of its size outside it (the fused walls lie ~1 % off); the ground and the podium beside stay
    t0 = time.perf_counter()
    height = float(np.ptp(gv[:, 2]))
    solid_v, solid_f, solid_info = solidify(gv, gf, yaw=yaw)
    solid_v, _ = snap_planes(solid_v, solid_f, tol=0.03)  # the voxel steps and the stubs carving left: flat again
    sdf = o3d.t.geometry.RaycastingScene()
    sdf.add_triangles(o3c.Tensor(solid_v.astype(np.float32)), o3c.Tensor(solid_f.astype(np.uint32)))
    cen = sv[sf].mean(1)
    size = float(np.ptp(gv[:, :2], 0).max())
    shell = sdf.compute_signed_distance(o3c.Tensor(cen.astype(np.float32))).numpy() < 0.02 * size
    # ... and what the fusion left standing on its roof (the subject owns its footprint's column)
    from scipy import ndimage

    cell = size / 200
    lo2 = solid_v[:, :2].min(0) - 12 * cell
    ij = np.floor((solid_v[:, :2] - lo2) / cell).astype(int)
    foot = np.zeros(tuple(ij.max(0)[::-1] + 13), bool)
    foot[ij[:, 1], ij[:, 0]] = True
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(foot, iterations=2))
    foot = ndimage.binary_dilation(foot, iterations=10)  # 5 % of its size round it: the fused roof edge's flaps
    cj = np.floor((cen[:, :2] - lo2) / cell).astype(int)
    inb = (cj >= 0).all(1) & (cj[:, 0] < foot.shape[1]) & (cj[:, 1] < foot.shape[0])
    on_foot = np.zeros(len(cen), bool)
    on_foot[inb] = foot[cj[inb, 1], cj[inb, 0]]
    shell |= on_foot & (cen[:, 2] > solid_v[:, 2].max() - 0.1 * height)
    keep_f = sf[~shell]
    t_solid = time.perf_counter() - t0

    # 3. the solid's atlas: the generated texture at the nearest point, then the photographs wherever a
    #    keyframe sees the solid (projected onto it directly: relayed through the generated mesh they smeared)
    t0 = time.perf_counter()
    corner_uv, gen_albedo, texels = transfer_texture(solid_v, solid_f, (gv, gf, guv, galb), size=TEXTURE_SIZE)
    solid_albedo, tex_info = retexture(to_sfm(solid_v), solid_f, None, gen_albedo, views, occluders=(to_sfm(sv), keep_f),
                                       sky=sky, texels=texels)  # fmt: skip
    del views
    torch.cuda.empty_cache()
    t_tex = time.perf_counter() - t0

    # 4. the files: the subject about its pivot (the centre of its base), and the scene
    out = run_dir / "export" / "complete"
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = solid_v.min(0), solid_v.max(0)
    fp_centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    local = solid_v - fp_centre
    files = _write_subject(local, solid_f, corner_uv, solid_albedo, out / "subject")
    solid = trimesh.Trimesh(local, solid_f, process=False)
    solid.export(out / "subject.stl")
    files.append(out / "subject.stl")
    fbx = write_fbx(files[0], out / "subject.fbx")
    if fbx:
        files.append(fbx)
    import io

    from PIL import Image

    s_scene = trimesh.load(str(mdir / "mesh_textured.glb"), process=False, maintain_order=True)
    s_mesh = next(iter(s_scene.geometry.values())) if isinstance(s_scene, trimesh.Scene) else s_scene
    s_mesh.update_faces(~shell)
    s_mesh.remove_unreferenced_vertices()
    buf = io.BytesIO()
    Image.fromarray(solid_albedo).save(buf, format="JPEG", quality=92)
    # the ground the flight did not see round the subject (within the flight's radius, twice)
    from drone3d.export.stage import _subject

    cams = np.array([im.projection_center() for im in posed])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in posed])
    sub = _subject(cams, axes)
    flight_r = fs * sub[1] if sub is not None else float(np.linalg.norm(np.ptp(fs * cams @ fr.T, 0)[:2])) / 2
    face_col = _face_colours(s_mesh)
    gfv, gff, gfc = fill_ground(sv, sf[~shell], face_col, fp_centre, 2.0 * flight_r, size / 100,
                                subject=(solid_v, solid_f))  # fmt: skip
    scene = trimesh.Scene()
    scene.add_geometry(s_mesh, node_name="measured_scene", geom_name="measured_scene")
    if len(gff):
        scene.add_geometry(trimesh.Trimesh(gfv @ Y_UP.T, gff, vertex_colors=gfc, process=False),
                           node_name="ground_fill", geom_name="ground_fill")  # fmt: skip
    pivot = np.eye(4)
    pivot[:3, 3] = Y_UP @ fp_centre
    scene.add_geometry(_textured(local, solid_f, corner_uv, Image.open(io.BytesIO(buf.getvalue()))),
                       node_name="subject", geom_name="subject", transform=pivot)  # fmt: skip
    scene.export(out / "scene.glb")
    files.append(out / "scene.glb")
    result = {
        "status": "ok", "model": model, "frame": frame.get("frame"), "units": frame.get("units"),
        "pivot_export_frame": fp_centre.round(5).tolist(), "height": round(height, 4),
        "generated_triangles": int(len(gf)), "base_slab_triangles": int((~slab_keep).sum()), "planes_snapped": planes,
        "carved_triangles": carved,
        "subject_triangles": int(len(solid_f)), "solid": solid_info, "watertight": bool(solid.is_watertight),
        "scene_triangles": int(len(keep_f)), "shell_triangles_replaced": int(shell.sum()),
        "ground_fill_triangles": int(len(gff)),
        "texture": tex_info, "generated_share": round(1 - tex_info["photo_texels"], 4),
        "note": "photographed where the keyframes see the object; elsewhere generated by TRELLIS.2 from one keyframe, not measured",
        "files": [str(p.relative_to(run_dir / "export")) for p in files],
        "seconds": {"carve": round(t_carve, 1), "texture": round(t_tex, 1), "solid": round(t_solid, 1), "total": round(time.perf_counter() - t_start, 1)},
    }  # fmt: skip
    (out / "result.json").write_text(json.dumps(result, indent=1))
    link_scene(run_dir, model)
    log.info("complete model: %d%% of the subject's texels photographed, solid watertight=%s, %.0f s",
             100 * tex_info["photo_texels"], solid.is_watertight, result["seconds"]["total"])
    return result


def _sky_gradient(images: list[np.ndarray], skies: list[np.ndarray], bands: int = 12) -> np.ndarray:
    """The photographed sky's mean colour per band of image rows -> [bands, 3] (missing bands: the nearest)."""
    acc, cnt = np.zeros((bands, 3)), np.zeros(bands)
    for img, sky in zip(images, skies, strict=True):
        h = sky.shape[0]
        small = img[:: max(1, img.shape[0] // h), :: max(1, img.shape[1] // sky.shape[1])][:h, : sky.shape[1]]
        rows = np.minimum((np.arange(h) * bands) // h, bands - 1)
        for b in range(bands):
            m = sky[rows == b]
            if m.any():
                acc[b] += small[rows == b][m].sum(0)
                cnt[b] += m.sum()
    have = np.flatnonzero(cnt > 0)
    if not len(have):
        return np.tile([[160.0, 180.0, 200.0]], (bands, 1))
    out = np.zeros((bands, 3))
    for b in range(bands):
        near = have[np.abs(have - b).argmin()]
        out[b] = acc[near] / cnt[near]
    return out


def _part_names(path: Path) -> list[str]:
    """The node names of a GLB's meshes, in ``render_mesh.load_parts`` order."""
    import trimesh

    scene = trimesh.load(str(path), process=False)
    return list(scene.graph.nodes_geometry) if isinstance(scene, trimesh.Scene) else ["mesh"]


def splats_360(run_dir: Path, *, model: int = 0, step_deg: float = 7.5, gap_deg: float = 20.0,
               rings: tuple[tuple[float, float], ...] = ((0.0, 1.0), (12.0, 1.15), (25.0, 1.3)), iterations: int | None = None,
               train: bool = True) -> dict:  # fmt: skip
    """Gaussian splats that are whole from every heading: trained on the keyframes and on views of the complete
    scene (``complete_model``) from the headings the flight never flew.

    A splat model is as complete as its views: trained on a 110-degree arc, the highrise's splats smear from
    everywhere else. Views rendered from the complete scene -- the flight's radius and elevation continued
    round the subject every ``step_deg`` (``rings``: extra elevation in degrees, radius factor; a ring 20
    degrees or more above the flight covers the flown headings too, where splats seen from above the flight
    hazed), looking at
    where the flight looked, the photographed sky behind -- supervise the rest. They are generated where the
    complete model is, and ``result.json`` lists them as such. Written to ``export/complete/splat_data`` (a
    COLMAP dataset: the keyframes, ``virtual/``, ``sparse/0``, ``masks/``: a view is supervised where the
    complete model is and in the sky above its horizon, never on the unmodelled ground below it) and
    ``splats/model_<N>_360``.
    """
    import shutil
    import tempfile

    import pycolmap
    import torch
    import trimesh
    from PIL import Image

    from drone3d.config import SplatConfig
    from drone3d.export.render_mesh import load_parts, render_parts
    from drone3d.export.stage import _subject
    from drone3d.splat.spirula import run_train

    t_start = time.perf_counter()
    out = run_dir / "export" / "complete"
    if not (out / "scene.glb").is_file():
        raise ValueError("no complete model: run complete_model first")
    sfm = json.loads((run_dir / "sfm" / "result.json").read_text())
    model_dir = Path(sfm["models"][model]["path"])
    name = model_dir.name
    frame = json.loads((run_dir / "export" / f"model_{name}" / "frame.json").read_text())
    fs, fr, ft = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])
    dataset = run_dir / "dataset"
    init = dataset / "sparse_init" / name
    rec = pycolmap.Reconstruction(str(init if (init / "images.bin").is_file() else model_dir))
    posed = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    cams = np.array([im.projection_center() for im in posed])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in posed])
    sub = _subject(cams, axes)
    if sub is None:
        raise ValueError("the flight circles no subject: nothing to complete round it")
    target = fs * fr @ sub[0] + ft
    ce = fs * cams @ fr.T + ft
    rel = ce - target
    radius = float(np.median(np.hypot(rel[:, 0], rel[:, 1])))
    elev = float(np.median(np.degrees(np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1])))))
    flown = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360
    cam0 = rec.cameras[posed[0].camera_id]
    w, h = int(cam0.width), int(cam0.height)
    fpx = float(cam0.params[0])
    # the headings the flight missed
    views = []
    for de, rf in rings:
        high = de >= 20.0  # well above the flight: no keyframe looks from there, so every heading, 15 degrees apart
        for az in np.arange(0, 360, 2 * step_deg if high else step_deg):
            if not high and (np.abs((flown - az + 180) % 360 - 180)).min() < gap_deg:
                continue
            a_ = np.radians(az)  # the flight's horizontal radius times rf, raised to its elevation plus de
            c = target + rf * radius * np.array([np.cos(a_), np.sin(a_), np.tan(np.radians(elev + de))])
            f_ = (target - c) / np.linalg.norm(target - c)
            right = np.cross(f_, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            views.append((float(az), de, c, np.stack([right, np.cross(f_, right), f_])))
    # the sky behind them: the photographs' sky gradient
    from torchvision.io import read_file

    from drone3d.gpu.nvjpeg import decode_jpeg

    pick = posed[:: max(1, len(posed) // 12)]
    imgs = [decode_jpeg(read_file(str(dataset / "images" / im.name)), device="cuda").permute(1, 2, 0).contiguous()
            for im in pick]  # fmt: skip

    class _V:  # the minimum _sky_masks reads
        def __init__(self, img):  # type: ignore[no-untyped-def]
            self.image = img

    skies = _sky_masks([_V(i) for i in imgs])
    grad = _sky_gradient([i.cpu().numpy() for i in imgs], skies)
    del imgs
    torch.cuda.empty_cache()
    parts = load_parts(out / "scene.glb")
    data = out / "splat_data"
    if data.exists():
        shutil.rmtree(data)
    (data / "images" / "virtual").mkdir(parents=True)
    (data / "masks" / "virtual").mkdir(parents=True)
    for d in sorted({im.name.split("/")[0] for im in posed}):  # the keyframes, as they are, every pixel supervised
        (data / "images" / d).symlink_to((dataset / "images" / d).resolve())
        (data / "masks" / d).mkdir(parents=True, exist_ok=True)
    white = Image.fromarray(np.full((h, w), 255, np.uint8))
    for im in posed:
        white.save(data / "masks" / Path(im.name).with_suffix(".png"))
    t0 = time.perf_counter()
    # a smooth gradient between the bands' centres (stepping from band to band showed as stripes)
    centres = (np.arange(len(grad)) + 0.5) * h / len(grad)
    column = np.stack([np.interp(np.arange(h) + 0.5, centres, grad[:, ch]) for ch in range(3)], 1)
    bg = np.broadcast_to(column[:, None, :], (h, w, 3)).round().astype(np.uint8)
    names = []
    names_in_glb = _part_names(out / "scene.glb")
    subject_part = names_in_glb.index("subject") if "subject" in names_in_glb else -2
    for k, (az, de, c, R) in enumerate(views):
        img, part = render_parts(parts, fpx, R, c, (w, h))
        hit = part >= 0
        img = np.where(hit[..., None], img, bg)
        nm = f"virtual/v{k:03d}_az{int(az):03d}_e{int(de):02d}.jpg"
        Image.fromarray(img).save(data / "images" / nm, quality=92)
        # supervised: the complete model, and the sky above the horizon; below it, where nothing was modelled,
        # the photographs see the distant city -- painting sky there made fog in front of the real views
        ys = np.arange(h)[:, None] + 0.5
        up = (R.T @ np.stack([np.zeros_like(ys[:, 0]), (ys[:, 0] - h / 2) / fpx, np.ones(h)]))[2] > 0  # per row
        keep = hit | up[:, None]
        if (np.abs((flown - az + 180) % 360 - 180)).min() < gap_deg:  # above the flight: the photographs have the
            keep = (part == subject_part) | (up[:, None] & ~hit)  # surroundings; the subject is what hazed there
        Image.fromarray((keep * 255).astype(np.uint8)).save(data / "masks" / Path(nm).with_suffix(".png"))
        names.append(nm)
    t_render = time.perf_counter() - t0
    # the COLMAP model: the keyframes' (with the dense cloud) plus a pinhole camera, rig, frame and image per view
    with tempfile.TemporaryDirectory() as tmp:
        rec.write_text(tmp)
        txt = {f: (Path(tmp) / f).read_text() for f in ("cameras.txt", "rigs.txt", "frames.txt", "images.txt", "points3D.txt")}
    cam_id = max(rec.cameras) + 1
    rig_id = max((int(line.split()[0]) for line in txt["rigs.txt"].splitlines() if line and not line.startswith("#")), default=0) + 1
    frame_id = max((int(line.split()[0]) for line in txt["frames.txt"].splitlines() if line and not line.startswith("#")), default=0) + 1
    img_id = max(rec.images) + 1
    txt["cameras.txt"] += f"{cam_id} PINHOLE {w} {h} {fpx} {fpx} {w / 2} {h / 2}\n"
    txt["rigs.txt"] += f"{rig_id} 1 CAMERA {cam_id}\n"
    for k, ((_, _, c, R), nm) in enumerate(zip(views, names, strict=True)):
        cs = fr.T @ ((c - ft) / fs)  # the camera in the SfM frame
        rs = R @ fr
        q = pycolmap.Rotation3d(rs).quat  # xyzw
        t = -rs @ cs
        pose = f"{q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]}"
        txt["frames.txt"] += f"{frame_id + k} {rig_id} {pose} 1 CAMERA {cam_id} {img_id + k}\n"
        txt["images.txt"] += f"{img_id + k} {pose} {cam_id} {nm}\n\n"
    # starting points on the complete subject and the filled ground, where the keyframes' cloud has none
    pts, cols = [], []
    for v_, f_, uv, alb in parts:
        m = trimesh.Trimesh(v_, f_, process=False)
        n_pts = min(200_000, max(1000, len(f_)))
        p, fi = trimesh.sample.sample_surface(m, n_pts, seed=0)
        if alb is None:  # vertex colours
            col = uv[f_[fi]].mean(1)
        else:
            tuv = uv[fi].mean(1)
            col = alb[np.clip(((1 - tuv[:, 1]) * alb.shape[0]).astype(int), 0, alb.shape[0] - 1),
                      np.clip((tuv[:, 0] * alb.shape[1]).astype(int), 0, alb.shape[1] - 1)]  # fmt: skip
        pts.append(p)
        cols.append(col)
    pts_s = ((np.concatenate(pts) - ft) / fs) @ fr
    cols_ = np.concatenate(cols).round().clip(0, 255).astype(int)
    pid = max(rec.points3D, default=0) + 1
    txt["points3D.txt"] += "".join(f"{pid + i} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0\n"
                                   for i, ((x, y, z), (r, g, b)) in enumerate(zip(pts_s, cols_, strict=True)))  # fmt: skip
    sp = data / "sparse" / "0"
    sp.mkdir(parents=True)
    for fname, body in txt.items():
        (sp / fname).write_text(body)
    merged = pycolmap.Reconstruction(str(sp))
    merged.write_binary(str(sp))
    for fname in txt:
        (sp / fname).unlink()
    result = {"status": "rendered", "views": len(views), "keyframes": len(posed), "radius": round(radius, 4),
              "elevation_deg": round(elev, 1), "headings_flown_deg": round(float(np.ptp(np.unwrap(np.radians(flown))) * 180 / np.pi), 1),
              "rings": [list(r) for r in rings], "step_deg": step_deg, "added_points": int(len(pts_s)),
              "note": "the views from unflown headings are renders of the complete model, whose unseen side is generated",
              "seconds": {"render": round(t_render, 1)}}  # fmt: skip
    if train:
        cfg = SplatConfig()
        ui = run_dir / "ui_config.yaml"
        if ui.is_file():
            import yaml

            for k_, v_ in ((yaml.safe_load(ui.read_text()) or {}).get("splat") or {}).items():
                if hasattr(cfg, k_):
                    setattr(cfg, k_, v_)
        tr = run_train(data, run_dir / "splats" / f"model_{name}_360", preset=cfg.preset, recon_dir="sparse/0",
                       iterations=iterations or cfg.iterations, quality=cfg.quality, depth_weight=0.0,
                       eval_interval=cfg.eval_interval,
                       flags={"train_resolution_divisor": cfg.resolution_divisor, "cache_images": cfg.cache_images,
                              "load_depths": 0, **cfg.flags})  # fmt: skip
        result.update(status="ok", **{k_: v_ for k_, v_ in tr.to_dict().items() if k_ not in ("config", "seconds")})
        result["seconds"]["train"] = round(tr.seconds, 1)
        if tr.splat_ply is not None:  # for the web viewer, in the model's export frame
            from drone3d.export.splats import compose, export_splat, train_from_world

            tf = compose((fs, fr, ft), train_from_world(tr.run_dir / "scene_transform.json"))
            result["web"] = export_splat(tr.splat_ply, out / "splats_360.splat", transform=tf)
            link_scene(run_dir, model)
    result["seconds"]["total"] = round(time.perf_counter() - t_start, 1)
    (out / "splats_360.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def link_scene(run_dir: Path, model: int = 0) -> dict | None:
    """Point the explorer's scene (export/scene.json, model ``model``: ``complete``) at the complete model's
    files that exist -> the entry, or None when there is no complete model."""
    out = run_dir / "export" / "complete"
    path = run_dir / "export" / "scene.json"
    if not (out / "scene.glb").is_file() or not path.is_file():
        return None
    entry = {"scene": "complete/scene.glb", "subject": "complete/subject.glb",
             "splat": "complete/splats_360.splat" if (out / "splats_360.splat").is_file() else None, "note": NOTE}  # fmt: skip
    scene = json.loads(path.read_text())
    scene["models"][model]["complete"] = entry
    path.write_text(json.dumps(scene, indent=1))
    return entry
