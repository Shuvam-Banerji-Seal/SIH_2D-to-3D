"""A clean model from the fused mesh: the ground as a terrain surface, what stands on it as smoothed objects.

The TSDF mesh is fused from 480 px depth maps: walls come out as crumpled facets (depth noise of a few voxels),
the ground as the same mesh with holes where no view reached, and the far field as streaks from grazing views.
Seen shaded -- or on a silhouette -- it reads as broken triangles. Aerial mapping has two better
representations, and a drone scene is both:

- **terrain**: the ground as a height field. The lowest surfaces per cell; a cell the opening by a window
  wider than any building cuts by more than an object's height (grey-scale erosion then dilation removes what
  stands up: a progressive morphological filter, Zhang et al. 2003) is not ground, and is inpainted from the
  ground around it; smoothed, triangulated as a regular grid: no streaks, no floaters, no holes;
- **objects**: the mesh where it stands above that terrain (or is steep), without small fragments, smoothed by
  bilateral normal filtering (Zheng et al. 2011: face normals averaged with neighbours of similar orientation,
  so a wall flattens and its corner stays sharp; then vertices moved to fit the filtered normals).

``clean_model(v, f)`` returns the combined mesh; the texture is baked on it as on the raw mesh.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bilateral_smooth", "clean_model", "terrain_surface"]


def _fill(grid: np.ndarray, valid: np.ndarray, *, iters: int = 400) -> np.ndarray:
    """Holes of a height grid filled by repeated neighbour averaging (a harmonic inpainting); valid cells fixed."""
    from scipy import ndimage

    g = np.where(valid, grid, 0.0).astype(np.float64)
    if not valid.any():
        return g
    # start from the nearest valid value, then relax: smooth, bounded by the surrounding values
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    g = g[tuple(idx)]
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float) / 4
    for _ in range(iters):
        g = np.where(valid, grid, ndimage.convolve(g, k, mode="nearest"))
    return g


def terrain_surface(points: np.ndarray, cell: float, window: float, *, rise: float | None = None, low: float = 15.0,
                    smooth_cells: float = 1.5, fill_iters: int = 400) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:  # fmt: skip
    """Ground height grid from surface points -> ``(z [H, W], data [H, W] bool, (x0, y0))``, cell centres at
    ``x0 + (j + .5) cell``.

    Per cell the ``low`` percentile of the heights (the ground under cars and eaves). A grey-scale opening by
    ``window`` removes whatever narrower than it stands up; a cell it lowers by more than ``rise`` (default 2
    cells) is not ground. Ground cells keep their own height -- the opening's floor would sit on the noise's
    minimum -- the rest are inpainted from them, and all smoothed over ``smooth_cells``. ``data`` marks cells
    that had points (the terrain is kept only near them).
    """
    from scipy import ndimage

    rise = 2.0 * cell if rise is None else rise
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    ij = np.floor((points[:, :2] - (x0, y0)) / cell).astype(np.int64)
    w, h = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    flat = ij[:, 1] * w + ij[:, 0]
    order = np.lexsort((points[:, 2], flat))
    fs, zs = flat[order], points[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(fs)) + 1]
    counts = np.diff(np.r_[starts, len(fs)])
    pick = starts + np.floor((counts - 1) * low / 100).astype(np.int64)
    z = np.full(h * w, np.nan)
    z[fs[pick]] = zs[pick]
    z = z.reshape(h, w)
    data = np.isfinite(z)
    filled = _fill(z, data, iters=0)  # nearest values: enough for the opening, and 2 s cheaper on a 1M-cell grid
    k = max(3, int(round(window / cell)) | 1)
    opened = ndimage.grey_dilation(ndimage.grey_erosion(filled, size=(k, k)), size=(k, k))
    ground = _fill(filled, data & (filled - opened <= rise), iters=fill_iters)
    ground = ndimage.gaussian_filter(ground, smooth_cells)
    return ground, data, (float(x0), float(y0))


def _face_adjacency(f: np.ndarray, n_v: int, neighbours: str = "vertex"):  # type: ignore[no-untyped-def]
    """Face-face pairs, both directions: faces sharing a vertex (``"vertex"``, a sparse incidence product, ~12
    per face) or an edge (``"edge"``, three per face, from sorted edges)."""
    if neighbours == "edge":
        import trimesh

        adj = trimesh.graph.face_adjacency(faces=np.asarray(f))
        return np.concatenate([adj[:, 0], adj[:, 1]]), np.concatenate([adj[:, 1], adj[:, 0]])
    from scipy import sparse

    rows = np.repeat(np.arange(len(f)), 3)
    inc = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, f.ravel())), shape=(len(f), n_v))
    adj = (inc @ inc.T).tocoo()
    keep = adj.row != adj.col
    return adj.row[keep], adj.col[keep]


def bilateral_smooth(v: np.ndarray, f: np.ndarray, *, normal_iters: int = 6, vertex_iters: int = 10,
                     sigma_r: float = 0.35, neighbours: str = "vertex") -> np.ndarray:  # fmt: skip
    """Bilateral normal filtering (Zheng et al. 2011) then the vertex update of Sun et al. 2007 -> new vertices.

    On the GPU when there is one: Colosseum's 431k object triangles, 5 M neighbour pairs, took 24 s in numpy
    (under 1 GB of GPU memory; if even that is not free, numpy).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return _bilateral_torch(v, f, normal_iters=normal_iters, vertex_iters=vertex_iters, sigma_r=sigma_r,
                                    neighbours=neighbours)
    except ImportError:
        pass
    except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is one
        import logging

        logging.getLogger(__name__).warning("bilateral smoothing on the CPU: %s", str(exc)[:120])
        torch.cuda.empty_cache()
    return _bilateral_numpy(v, f, normal_iters=normal_iters, vertex_iters=vertex_iters, sigma_r=sigma_r,
                            neighbours=neighbours)


def _bilateral_torch(v: np.ndarray, f: np.ndarray, *, normal_iters: int, vertex_iters: int, sigma_r: float,
                     neighbours: str = "vertex") -> np.ndarray:  # fmt: skip
    import torch

    dev = "cuda"
    a_, b_ = _face_adjacency(np.asarray(f, np.int64), len(v), neighbours)
    fa = torch.as_tensor(np.asarray(f, np.int64), device=dev)
    A, B = torch.as_tensor(a_, device=dev, dtype=torch.long), torch.as_tensor(b_, device=dev, dtype=torch.long)
    x = torch.as_tensor(np.asarray(v, np.float64), device=dev)
    for _ in range(max(1, vertex_iters // 5)):
        tri = x[fa]
        cen = tri.mean(1)
        cr = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * cr.norm(dim=1)
        n = cr / (2 * area).clamp_min(1e-12)[:, None]
        dc = cen[A] - cen[B]
        sigma_s = float(dc.norm(dim=1).median()) or 1.0
        ws = torch.exp(-(dc**2).sum(1) / (2 * sigma_s**2)) * area[B]
        for _ in range(normal_iters):
            wt = ws * torch.exp(-((n[A] - n[B]) ** 2).sum(1) / (2 * sigma_r**2))
            acc = (n * area[:, None]).index_add_(0, A, wt[:, None] * n[B])
            n = acc / acc.norm(dim=1).clamp_min(1e-12)[:, None]
        for _ in range(5):  # vertices moved onto the planes of their filtered faces
            tri = x[fa]
            cen = tri.mean(1)
            move = torch.zeros_like(x)
            cnt = torch.zeros(len(x), device=dev, dtype=x.dtype)
            for c in range(3):
                d = ((n * (cen - tri[:, c])).sum(1))[:, None] * n
                move.index_add_(0, fa[:, c], d)
                cnt.index_add_(0, fa[:, c], torch.ones(len(fa), device=dev, dtype=x.dtype))
            x = x + move / cnt.clamp_min(1)[:, None]
    out = x.cpu().numpy()
    del x, A, B, fa
    torch.cuda.empty_cache()
    return out


def _bilateral_numpy(v: np.ndarray, f: np.ndarray, *, normal_iters: int, vertex_iters: int, sigma_r: float,
                     neighbours: str = "vertex") -> np.ndarray:  # fmt: skip
    v = np.asarray(v, np.float64).copy()
    f = np.asarray(f, np.int64)
    a_, b_ = _face_adjacency(f, len(v), neighbours)
    for _ in range(max(1, vertex_iters // 5)):
        tri = v[f]
        cen = tri.mean(1)
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(cr, axis=1)
        n = cr / np.maximum(2 * area, 1e-12)[:, None]
        sigma_s = float(np.median(np.linalg.norm(cen[a_] - cen[b_], axis=1))) or 1.0
        for _ in range(normal_iters):
            ws = np.exp(-np.sum((cen[a_] - cen[b_]) ** 2, 1) / (2 * sigma_s**2))
            wr = np.exp(-np.sum((n[a_] - n[b_]) ** 2, 1) / (2 * sigma_r**2))
            wt = area[b_] * ws * wr
            acc = n * area[:, None]  # the face itself
            np.add.at(acc, a_, wt[:, None] * n[b_])
            n = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-12)[:, None]
        for _ in range(5):  # vertices moved onto the planes of their filtered faces
            tri = v[f]
            cen = tri.mean(1)
            move = np.zeros_like(v)
            cnt = np.zeros(len(v))
            for c in range(3):
                d = np.sum(n * (cen - tri[:, c]), 1)[:, None] * n
                np.add.at(move, f[:, c], d)
                np.add.at(cnt, f[:, c], 1)
            v += move / np.maximum(cnt, 1)[:, None]
    return v


def clean_model(v: np.ndarray, f: np.ndarray, *, voxel: float, window: float, keep: float = 0.002,
                height: float | None = None, grid_cells: int = 1_000_000, fill_iters: int = 60,
                neighbours: str = "edge") -> tuple[np.ndarray, np.ndarray, dict]:  # fmt: skip
    """Terrain + smoothed objects from a fused mesh (z up) -> ``(vertices, faces, info)``.

    ``window``: wider than the widest building (the opening removes what is narrower); ``height``: how far above
    the terrain a surface must stand to be an object (default 3 voxels); ``keep``: object fragments with fewer
    than this share of the object triangles are dropped (floaters). ``fill_iters`` (60) and ``neighbours``
    (``"edge"``) against 400 and ``"vertex"``: 0.8 s instead of 3.7 on rural's model, the result a median 0.00
    to 0.02 voxels from it (p95 0.19-0.27) on rural and Angkor Wat.
    """
    import trimesh

    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    ext = np.ptp(v[:, :2], axis=0)
    cell = max(2.0 * voxel, float(np.sqrt(ext[0] * ext[1] / grid_cells)))
    height = 3.0 * voxel if height is None else height
    ground, data, (x0, y0) = terrain_surface(v, cell, window, rise=height, fill_iters=fill_iters)
    gh, gw = ground.shape

    def ground_at(xy: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        return map_coordinates(ground, [(xy[:, 1] - y0) / cell - 0.5, (xy[:, 0] - x0) / cell - 0.5], order=1, mode="nearest")

    tri = v[f]
    cen = tri.mean(1)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz = np.abs(cr[:, 2]) / np.maximum(np.linalg.norm(cr, axis=1), 1e-12)
    above = cen[:, 2] - ground_at(cen[:, :2])
    obj = (above > height) | ((nz < 0.6) & (above > 0.3 * height))  # standing up, or a steep face off the ground
    of = f[obj]
    # object fragments: connected components below `keep` of the object triangles are dropped
    parts = trimesh.Trimesh(vertices=v, faces=of, process=False)
    comps = trimesh.graph.connected_components(parts.face_adjacency, nodes=np.arange(len(of)), min_len=1)
    big = [c for c in comps if len(c) >= keep * len(of)]
    of = of[np.concatenate(big)] if big else of[:0]
    used, inv = np.unique(of, return_inverse=True)
    ov = bilateral_smooth(v[used], inv.reshape(-1, 3), neighbours=neighbours) if len(of) else np.zeros((0, 3))
    of = inv.reshape(-1, 3)
    # terrain grid, only where there were points (dilated a little, so holes between them are closed)
    from scipy import ndimage

    near = ndimage.binary_closing(ndimage.binary_dilation(data, iterations=2), iterations=3)
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    tv = np.stack([x0 + (jj + 0.5) * cell, y0 + (ii + 0.5) * cell, ground], -1).reshape(-1, 3)
    q = np.arange(gh * gw).reshape(gh, gw)
    a, b, c_, d = q[:-1, :-1], q[:-1, 1:], q[1:, :-1], q[1:, 1:]
    ok = (near[:-1, :-1] & near[:-1, 1:] & near[1:, :-1] & near[1:, 1:]).ravel()
    tf = np.concatenate([np.stack([a.ravel(), b.ravel(), d.ravel()], 1)[ok], np.stack([a.ravel(), d.ravel(), c_.ravel()], 1)[ok]])
    tused, tinv = np.unique(tf, return_inverse=True)
    tv, tf = tv[tused], tinv.reshape(-1, 3)
    out_v = np.concatenate([tv, ov])
    out_f = np.concatenate([tf, of + len(tv)])
    info = {"terrain_triangles": int(len(tf)), "object_triangles": int(len(of)), "cell": round(cell, 5),
            "window": round(window, 5), "object_fragments_dropped": int(len(comps) - len(big)),
            "raw_triangles": int(len(f))}  # fmt: skip
    return out_v, out_f, info
