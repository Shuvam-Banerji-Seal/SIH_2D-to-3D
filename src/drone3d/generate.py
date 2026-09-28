"""A generated object beside the measured model: TRELLIS.2 on the keyframe that shows the building whole.

The measured model (flow SfM, fused depth) is only as complete as the flight: the far side of a building
the drone never circled is missing, a short shot leaves gaps. TRELLIS.2 (microsoft/TRELLIS.2-4B) generates a
complete, textured object from one image. It is *not* a measurement -- the sides the keyframe does not
show are invented and the scale is arbitrary -- so it is written apart (``export/generated/``), listed in
the model catalog as generated, and never enters the deliverables or the metrics.

It runs in TRELLIS.2's own environment (``tools/setup_trellis2.sh``: torch 2.7, its CUDA extensions) as a
subprocess of ``tools/trellis2_generate.py``: ~2.5 min to load the weights, ~80 s to generate at 1024^3 and
~80 s to bake the GLB on the A100, 8 GB of GPU memory at most.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "third_party" / "TRELLIS.2" / ".venv" / "bin" / "python"
SCRIPT = ROOT / "tools" / "trellis2_generate.py"
NOTE = "generated from one keyframe by TRELLIS.2, not measured: the sides the flight never saw are invented"
LICENSES = {"microsoft/TRELLIS.2-4B": "MIT", "briaai/RMBG-2.0": "bria-rmbg-2.0 (non-commercial)",
            "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3-license"}  # fmt: skip


def available() -> bool:
    """TRELLIS.2's environment and the wrapper script are installed."""
    return PYTHON.is_file() and SCRIPT.is_file()


def subject_keyframe(cams: np.ndarray, axes: np.ndarray, *, cone_deg: float = 12.0) -> int:
    """Index of the keyframe that shows the subject whole.

    The subject is what the optical axes converge on (:func:`drone3d.export.stage._subject`); of the
    keyframes looking at it within ``cone_deg``, the farthest from it frames it whole -- on Colosseum's
    merged model, a view of the entire amphitheatre, where the nearest showed one arch. Axes that hardly
    converge (a survey, a fly-by) give the middle keyframe.
    """
    from drone3d.export.stage import _subject

    mid = len(cams) // 2
    sub = _subject(cams, axes)
    if sub is None:
        return mid
    rel = sub[0] - cams
    dist = np.linalg.norm(rel, axis=1)
    looking = np.einsum("ij,ij->i", rel, axes) / np.maximum(dist, 1e-12) > np.cos(np.radians(cone_deg))
    if not looking.any():
        return mid
    return int(np.flatnonzero(looking)[np.argmax(dist[looking])])


def pick_keyframe(run_dir: Path) -> tuple[Path, dict]:
    """The largest model's subject keyframe -> ``(image path, info)``."""
    import pycolmap

    sfm = json.loads((run_dir / "sfm" / "result.json").read_text())
    models = sorted(sfm.get("models", []), key=lambda m: -(m.get("images") or 0))
    if not models:
        raise ValueError("no SfM model: run the pipeline first")
    rec = pycolmap.Reconstruction(models[0]["path"])
    posed = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if im.has_pose]
    cams = np.array([im.projection_center() for im in posed])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in posed])
    k = subject_keyframe(cams, axes)
    return run_dir / "dataset" / "images" / posed[k].name, {"model": Path(models[0]["path"]).name, "keyframe": posed[k].name}


def subject_crop(run_dir: Path, image: Path, keyframe: str, *, model: int = 0, margin: float = 0.12,
                 footprint: float = 0.3) -> Path | None:  # fmt: skip
    """The keyframe cropped to the subject, from the measured model itself -> the crop's path, or None.

    The subject's surface (within ``footprint`` x the subject radius -- the cameras' distance -- of it) is projected
    into the keyframe and the image cut to that box plus ``margin``: the generator sees the building, not the city. On a
    wide keyframe of a highrise orbit the background remover alone kept half the city, and TRELLIS.2 generated
    a city block with the tower small in it.
    """
    import pycolmap
    import trimesh
    from PIL import Image

    scene_models = json.loads((run_dir / "export" / "scene.json").read_text())["models"]
    sm = scene_models[model]
    if not sm.get("focus"):
        return None
    name = sm["dir"].split("_", 1)[1]
    frame = json.loads((run_dir / "export" / sm["dir"] / "frame.json").read_text())
    fs, fr, ft = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])
    c, r = np.asarray(sm["focus"]["center"]), float(sm["focus"]["radius"])
    v = np.asarray(trimesh.load(run_dir / "export" / sm["dir"] / "mesh.ply", force="mesh", process=False).vertices)
    near = np.hypot(v[:, 0] - c[0], v[:, 1] - c[1]) < footprint * r
    if near.sum() < 200:
        return None
    ground = float(np.percentile(v[near, 2], 5))
    top = float(np.percentile(v[near, 2], 99))
    # the subject is what rises above its surroundings there: its upper half fixes the box's sides, the box
    # runs from the subject's top down to the ground
    tall = v[near & (v[:, 2] > ground + 0.5 * (top - ground))]
    body = np.concatenate([tall, np.c_[tall[:, :2], np.full(len(tall), ground)]]) if len(tall) >= 100 else v[near]
    rec = pycolmap.Reconstruction(str(run_dir / "dataset" / "sparse" / name))
    im = next((i for i in rec.images.values() if i.name == keyframe), None)
    if im is None:
        return None
    cam = rec.cameras[im.camera_id]
    pose = im.cam_from_world()
    x_sfm = ((body - ft) / fs) @ fr  # export frame -> SfM frame (fr is a rotation)
    pc = x_sfm @ np.asarray(pose.rotation.matrix()).T + np.asarray(pose.translation)
    pc = pc[pc[:, 2] > 1e-6]
    img = Image.open(image)
    sx = img.width / cam.width
    u = (cam.params[0] * pc[:, 0] / pc[:, 2] + cam.params[1]) * sx
    w_ = (cam.params[0] * pc[:, 1] / pc[:, 2] + cam.params[2]) * sx
    x0, x1 = np.percentile(u, [1, 99])
    y0, y1 = np.percentile(w_, [1, 99])
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    box = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(img.width, int(x1 + mx)), min(img.height, int(y1 + my)))
    if box[2] - box[0] < 64 or box[3] - box[1] < 64:
        return None
    out = out_dir(run_dir) / "subject_crop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.crop(box).save(out)
    return out


def out_dir(run_dir: Path) -> Path:
    return run_dir / "export" / "generated"


def status(run_dir: Path) -> dict | None:
    f = out_dir(run_dir) / "result.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def generate_object(run_dir: Path, *, image: Path | None = None, resolution: str = "1024", seed: int = 0,
                    timeout_s: float = 1800.0) -> dict:  # fmt: skip
    """Generate ``export/generated/object.glb`` for a finished run -> the result record (also result.json)."""
    if not available():
        raise RuntimeError(f"TRELLIS.2 is not installed ({PYTHON}); run tools/setup_trellis2.sh")
    out = out_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"keyframe": None}
    if image is None:
        image, info = pick_keyframe(run_dir)
        try:  # cut to the subject with the measured model (None: no subject, or it is not in view)
            crop = subject_crop(run_dir, image, info["keyframe"], model=int(info["model"]))
        except Exception as exc:  # noqa: BLE001 -- the whole keyframe still works
            log.warning("subject crop failed: %s", exc)
            crop = None
        if crop is not None:
            image, info = crop, {**info, "cropped_to_subject": True}
    record = {"status": "running", "started": time.time(), "pid": os.getpid(), "image": str(image), **info,
              "resolution": resolution, "seed": seed, "generator": "microsoft/TRELLIS.2-4B", "note": NOTE,
              "licenses": LICENSES}  # fmt: skip
    (out / "result.json").write_text(json.dumps(record, indent=1))
    glb = out / "object.glb"
    # the shell's OpenMP binding pins a child to one core (see drone3d.__init__)
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([str(PYTHON), str(SCRIPT), str(glb), str(image), "--res", resolution, "--seed", str(seed)],
                              capture_output=True, text=True, env=env, timeout=timeout_s, cwd=ROOT)  # fmt: skip
        ok = proc.returncode == 0 and glb.is_file()
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    except subprocess.TimeoutExpired:
        ok, tail = False, [f"timed out after {timeout_s:.0f} s"]
    record.update(status="ok" if ok else "failed", seconds=round(time.perf_counter() - t0, 1), finished=time.time(),
                  glb="object.glb" if ok else None, input="object.input.png" if (out / "object.input.png").is_file() else None,
                  log=tail)  # fmt: skip
    if ok:  # placed where the subject stands, so the explorer can fill the model's unseen sides with it
        try:
            record["aligned"] = align_to_model(run_dir, model=int(info.get("model") or 0))
            _link_scene(run_dir, int(info.get("model") or 0))
        except (ValueError, FileNotFoundError, KeyError) as exc:  # no subject, too little surface: unplaced
            record["aligned"] = {"status": "skipped", "reason": str(exc)[:200]}
    (out / "result.json").write_text(json.dumps(record, indent=1))
    if not ok:
        log.warning("generate %s failed: %s", run_dir.name, " | ".join(tail[-3:]))
    return record


def _link_scene(run_dir: Path, model: int) -> None:
    """Point the explorer's scene at the placed object (export/scene.json, model ``model``: ``generated``)."""
    path = run_dir / "export" / "scene.json"
    scene = json.loads(path.read_text())
    scene["models"][model]["generated"] = {"mesh": "generated/object_aligned.glb", "note": NOTE}
    path.write_text(json.dumps(scene, indent=1))


def _rz(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _upright_icp(gp: np.ndarray, bp: np.ndarray, yaw: float, scale: float, t: np.ndarray, thr: float,
                 iters: int = 40) -> tuple[float, float, np.ndarray]:  # fmt: skip
    """ICP for heading, scale and translation only: the object stays upright, as generated and as the levelled
    model is. Each measured point pairs with its nearest generated point within ``thr`` (the measurement is the
    partial side); the 4-DoF similarity then has a closed form -- the heading from the horizontal covariance."""
    from scipy.spatial import cKDTree

    for _ in range(iters):
        moved = scale * gp @ _rz(yaw).T + t
        d, j = cKDTree(moved).query(bp)
        ok = d < thr
        if ok.sum() < 20:
            break
        g, b = gp[j[ok]], bp[ok]
        gm, bm = g.mean(0), b.mean(0)
        gc, bc = g - gm, b - bm
        yaw = float(np.arctan2((gc[:, 0] * bc[:, 1] - gc[:, 1] * bc[:, 0]).sum(), (gc[:, 0] * bc[:, 0] + gc[:, 1] * bc[:, 1]).sum()))
        rg = gc @ _rz(yaw).T
        scale = float((rg * bc).sum() / max((gc**2).sum(), 1e-12))
        t = bm - scale * _rz(yaw) @ gm
    return yaw, scale, t


def align_to_model(run_dir: Path, *, model: int = 0, samples: int = 40000, yaws: int = 24) -> dict:
    """Place the generated object in the measured model's frame -> record (also ``export/generated/aligned.json``).

    The generated object has its own frame and scale. It is matched to the measured surface near the subject
    (above the ground, within half a subject radius of it): the scale from the two heights, the yaw by trying
    ``yaws`` headings, each refined by an upright ICP (heading, scale and position only); the best cover wins. The result,
    ``object_aligned.glb``, sits where the building stands, so the parts the flight never saw (a half orbit
    leaves the back empty) come from the generated object -- shown as generated, never measured.
    """
    import trimesh
    from scipy.spatial import cKDTree

    out = out_dir(run_dir)
    scene = json.loads((run_dir / "export" / "scene.json").read_text())["models"][model]
    focus = scene.get("focus")
    if not focus:
        raise ValueError("the model has no subject to align the generated object to")
    c, r = np.asarray(focus["center"]), float(focus["radius"])
    meas = trimesh.load(run_dir / "export" / f"model_{model}" / "mesh.ply", force="mesh", process=False)
    mv = np.asarray(meas.vertices)
    near = (np.hypot(mv[:, 0] - c[0], mv[:, 1] - c[1]) < 0.5 * r)
    ground = float(np.percentile(mv[near, 2], 5)) if near.any() else float(np.percentile(mv[:, 2], 5))
    body = mv[near & (mv[:, 2] > ground + 0.05 * r)]
    if len(body) < 500:
        raise ValueError("too little measured surface near the subject")
    gen = trimesh.load(out / "object.glb", force="mesh", process=False)
    gv = np.asarray(gen.vertices) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T  # glTF y-up -> z-up
    gen.vertices = gv
    gp = np.asarray(trimesh.sample.sample_surface(gen, samples, seed=0)[0])  # seeded: the same fit every time
    rng = np.random.default_rng(0)
    bp = body[rng.choice(len(body), min(samples, len(body)), replace=False)]
    s0 = (np.percentile(bp[:, 2], 98) - ground) / max(np.percentile(gp[:, 2], 98) - gp[:, 2].min(), 1e-9)
    thr = 0.04 * r
    base = np.array([np.median(gp[:, 0]), np.median(gp[:, 1]), gp[:, 2].min()])

    def cover(yaw: float, scale: float, t: np.ndarray) -> tuple[float, float]:
        d, _ = cKDTree(scale * gp @ _rz(yaw).T + t).query(bp)
        return float((d < thr).mean()), float(np.median(d))

    best = None
    for k in range(yaws):
        a0 = 2 * np.pi * k / yaws
        t0 = np.array([c[0], c[1], ground]) - s0 * _rz(a0) @ base
        yaw, sc, t = _upright_icp(gp, bp, a0, s0, t0, thr)
        score, med = cover(yaw, sc, t)
        if best is None or score > best[0]:
            best = (score, yaw, sc, t, med)
    # ICP with scale on a half-seen subject shrinks towards the measured side (-5 % on a synthetic tower); the ground-
    # to-roof height is steadier where the roof was seen: refit the position at that scale and keep the better cover
    _, yaw, sc, t, _ = best
    gz = gp @ _rz(yaw).T
    s_h = (np.percentile(bp[:, 2], 98) - ground) / max(np.percentile(gz[:, 2], 98) - gz[:, 2].min(), 1e-9)
    y2, _, t2 = _upright_icp(gp, bp, yaw, s_h, t + (sc - s_h) * _rz(yaw) @ base, thr, iters=1)  # one step: t at s_h
    for _ in range(20):  # translation and heading only, at the height's scale
        moved = s_h * gp @ _rz(y2).T + t2
        d, j = cKDTree(moved).query(bp)
        ok = d < thr
        if ok.sum() < 20:
            break
        t2 = t2 + (bp[ok] - moved[j[ok]]).mean(0)
    score_h, med_h = cover(y2, s_h, t2)
    if score_h >= best[0] - 0.02:  # as good a cover: the height's scale
        best = (score_h, y2, s_h, t2, med_h)
    score, yaw, sc, tt, med = best
    t = np.eye(4)
    t[:3, :3] = sc * _rz(yaw)
    t[:3, 3] = tt
    best = (score, t, med)
    score, t, med = best
    t = t.copy()
    t[2, 3] += ground - float((gp @ t[:3, :3].T + t[:3, 3])[:, 2].min())  # it stands on the measured ground
    pv = gv @ t[:3, :3].T + t[:3, 3]
    faces = np.asarray(gen.faces)
    # trimmed to the subject's measured footprint: what stands above its surroundings near the subject, its box
    # grown by 10 % -- the generator adds a base of its own (Jal Mahal: a disc of lake over the measured water;
    # the highrise: a block beside it); two facades of a half orbit already span the whole footprint
    top = float(np.percentile(body[:, 2], 99))
    tall = body[body[:, 2] > ground + 0.5 * (top - ground)]
    trimmed = 0
    if len(tall) >= 200:
        lo, hi = tall[:, :2].min(0), tall[:, :2].max(0)
        grow = 0.1 * (hi - lo) + 0.01 * r
        fc = pv[faces].mean(1)
        inside = np.all((fc[:, :2] >= lo - grow) & (fc[:, :2] <= hi + grow), axis=1)
        trimmed = int((~inside).sum())
        faces = faces[inside]
    # and without the flat slab it stands on (Jal Mahal's: water): near-horizontal faces in its lowest 3 %
    tri = pv[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flat = np.abs(n[:, 2]) > 0.8 * np.linalg.norm(n, axis=1).clip(1e-12)
    z0, z1 = float(pv[:, 2].min()), float(pv[:, 2].max())
    slab = flat & (tri[:, :, 2].mean(1) < z0 + 0.03 * (z1 - z0))
    trimmed += int(slab.sum())
    faces = faces[~slab]
    placed = trimesh.Trimesh(vertices=pv, faces=faces, visual=gen.visual, process=False)
    placed.remove_unreferenced_vertices()
    placed.vertices = np.asarray(placed.vertices) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T  # back to y-up
    placed.export(out / "object_aligned.glb")
    rec = {"model": model, "transform_zup": t.round(6).tolist(), "measured_covered": round(score, 3),
           "median_gap_rel": round(med / r, 4), "scale": round(float(np.cbrt(abs(np.linalg.det(t[:3, :3])))), 5),
           "trimmed_faces": trimmed,
           "glb": "object_aligned.glb"}  # fmt: skip
    (out / "aligned.json").write_text(json.dumps(rec, indent=1))
    return rec
