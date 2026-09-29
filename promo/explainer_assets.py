"""Real-run footage for the explainer film (promo/explainer.template.html), every clip from our own runs.

    uv run python promo/explainer_assets.py [clip ...]      # all clips by default; see CLIPS

* ``fly_<name>.mp4`` -- a run's clean textured model flown along the drone's own path (the keyframe poses,
  Catmull-Rom + slerp), pulled back a little for context: the camera only looks where the flight looked, so
  what it shows was captured -- full terrain, no open backs;
* ``turn_complete.mp4`` -- the highrise's complete model (drone3d.complete) circled 360 degrees;
* ``turn_splats.mp4`` -- its 360-degree Gaussian splats circled the same way;
* ``clean_wipe.mp4`` -- the fused mesh against the clean model, shaded, a wipe across the frame.

Rendering is ray casting (Open3D, the CPU) for meshes and gsplat for splats; the background is the film's
charcoal, graded, so the clips sit in its screens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)

BUILD = ROOT / "promo" / "build" / "explainer"
OUT = ROOT / "outputs"
BG_TOP, BG_BOT = np.array([34, 39, 46]), np.array([14, 16, 19])  # the film's charcoal, graded


def _frame(run: Path, k: int = 0):  # type: ignore[no-untyped-def]
    sfm = json.loads((run / "sfm" / "result.json").read_text())
    model_dir = Path(sfm["models"][k]["path"])
    fr_ = json.loads((run / "export" / f"model_{model_dir.name}" / "frame.json").read_text())
    return model_dir, float(fr_["scale"]), np.asarray(fr_["rotation"]), np.asarray(fr_["translation"])


def _poses(run: Path, k: int = 0):  # type: ignore[no-untyped-def]
    """Keyframe cameras in the export frame -> (centres [N, 3], cam_from_world rotations [N, 3, 3], f, w, h)."""
    import pycolmap

    model_dir, s, r, t = _frame(run, k)
    rec = pycolmap.Reconstruction(str(model_dir))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    c = np.array([s * r @ im.projection_center() + t for im in ims])
    rot = np.array([np.asarray(im.cam_from_world().rotation.matrix()) @ r.T for im in ims])
    cam = rec.cameras[ims[0].camera_id]
    return c, rot, float(cam.params[0]), int(cam.width), int(cam.height)


def _background(h: int, w: int) -> np.ndarray:
    a = np.linspace(0, 1, h)[:, None, None]
    return np.broadcast_to((BG_TOP * (1 - a) + BG_BOT * a).round().astype(np.uint8), (h, w, 3)).copy()


def _encode(frames: list[np.ndarray] | None, out: Path, fps: int = 30, pattern: Path | None = None) -> Path:
    from drone3d.io.nvdec import ffmpeg_bin

    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-crf", "15", "-preset", "slow",
           "-pix_fmt", "yuv420p", str(out)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def _look(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def _subject(run: Path, k: int = 0) -> tuple[np.ndarray, float]:
    from drone3d.export.stage import _subject as subject_of

    c, rot, _, _, _ = _poses(run, k)
    axes = rot[:, 2]
    sub = subject_of(c, axes)
    if sub is None:
        return c.mean(0), float(np.median(np.linalg.norm(c - c.mean(0), axis=1)))
    return sub


def _render_mesh(parts, f: float, R: np.ndarray, c: np.ndarray, size: tuple[int, int]) -> np.ndarray:  # type: ignore[no-untyped-def]
    from drone3d.export.render_mesh import render_parts

    img, part = render_parts(parts, f, R, c, size)
    bg = _background(size[1], size[0])
    return np.where((part >= 0)[..., None], img, bg)


def fly(run: Path, out: Path, *, model: int = 0, seconds: float = 8.0, size=(1920, 1080), back: float = 0.25,
        start: float = 0.0, span: float = 1.0, mesh: Path | None = None) -> Path:  # fmt: skip
    """The textured clean model along the flight (the part ``start`` .. ``start + span`` of it), each camera
    pulled back ``back`` x the median camera-to-subject distance along its own axis."""
    from drone3d.export.render_mesh import load_parts
    from drone3d.splat.render import camera_path

    c, rot, f, w0, _ = _poses(run, model)
    lo, hi = int(start * (len(c) - 1)), max(int((start + span) * (len(c) - 1)), int(start * (len(c) - 1)) + 2)
    c, rot = c[lo : hi + 1], rot[lo : hi + 1]
    mdir = run / "export" / f"model_{_frame(run, model)[0].name}"
    parts = load_parts(mesh or mdir / "mesh_textured.glb")
    _, radius = _subject(run, model)
    n = int(seconds * 30)
    pc, pr = camera_path(c, rot, n)
    fpx = f * size[0] / w0
    frames = []
    for cc, rr in zip(pc, pr, strict=True):
        eye = cc - back * radius * rr[2]  # back along the view axis
        frames.append(_render_mesh(parts, fpx, rr, eye, size))
    return _encode(frames, out)


def target_z(run: Path, glb: Path, lift: float, model: int = 0) -> float | None:
    """The height ``lift`` of the way up what stands at the subject (``turntable``'s aim), for the splats' too."""
    from drone3d.export.render_mesh import load_parts

    target, radius = _subject(run, model)
    v = np.concatenate([p[0] for p in load_parts(glb)])
    near = np.linalg.norm(v[:, :2] - target[:2], axis=1) < 0.5 * radius
    if near.sum() <= 100:
        return None
    z0, z1 = np.percentile(v[near, 2], [2, 99.5])
    return float(z0 + lift * (z1 - z0))


def turntable(run: Path, glb: Path, out: Path, *, model: int = 0, seconds: float = 12.0, size=(1920, 1080),
              elevation: float = 18.0, distance: float = 1.35, turns: float = 1.0, lift: float = 0.5) -> Path:  # fmt: skip
    """``glb`` (export frame, y-up) circled round the run's subject from the first keyframe's heading."""
    from drone3d.export.render_mesh import load_parts

    parts = load_parts(glb)
    target, radius = _subject(run, model)
    v = np.concatenate([p[0] for p in parts])
    near = np.linalg.norm(v[:, :2] - target[:2], axis=1) < 0.5 * radius
    if near.sum() > 100:
        z0, z1 = np.percentile(v[near, 2], [2, 99.5])
        target = np.array([target[0], target[1], z0 + lift * (z1 - z0)])
    c, _, f, w0, _ = _poses(run, model)
    a0 = np.arctan2(c[0, 1] - target[1], c[0, 0] - target[0])
    fpx = f * size[0] / w0
    frames = []
    n = int(seconds * 30)
    for i in range(n):
        a = a0 + 2 * np.pi * turns * i / n
        e = np.radians(elevation)
        eye = target + distance * radius * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        frames.append(_render_mesh(parts, fpx, _look(eye, target), eye, size))
    return _encode(frames, out)


def splat_turntable(run: Path, splat_dir: Path, out: Path, *, model: int = 0, seconds: float = 12.0, size=(1920, 1080),
                    elevation: float = 18.0, distance: float = 1.35, turns: float = 1.0, target_z: float | None = None) -> Path:  # fmt: skip
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")
    import gsplat
    import torch

    from drone3d.splat.render import load_splat_ply

    _, s, r, t = _frame(run, model)
    ply = sorted(splat_dir.rglob("splat*.ply"), key=lambda p: p.stat().st_mtime)[-1]
    spl = load_splat_ply(ply)
    tf = json.loads((splat_dir / "scene_transform.json").read_text())["train_from_world"]
    s_tw, r_tw, t_tw = float(tf["scale"]), np.array(tf["rotation"]["matrix_3x3"]), np.array(tf["translation"])
    target, radius = _subject(run, model)
    if target_z is not None:
        target = np.array([target[0], target[1], target_z])
    c, _, f, w0, _ = _poses(run, model)
    a0 = np.arctan2(c[0, 1] - target[1], c[0, 0] - target[0])
    fpx = f * size[0] / w0
    K = torch.tensor([[fpx, 0, size[0] / 2], [0, fpx, size[1] / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
    frames = []
    n = int(seconds * 30)
    for i in range(n):
        a = a0 + 2 * np.pi * turns * i / n
        e = np.radians(elevation)
        eye = target + distance * radius * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        R = _look(eye, target)
        cs = r.T @ ((eye - t) / s)  # export -> SfM -> training frame
        rs = R @ r
        rc = rs @ r_tw.T
        vm = np.eye(4)
        vm[:3, :3] = rc
        vm[:3, 3] = s_tw * (-rs @ cs) - rc @ t_tw
        with torch.no_grad():
            img, alpha, _ = gsplat.rasterization(spl["means"], spl["quats"], spl["scales"], spl["opacities"], spl["sh"],
                                                 torch.tensor(vm[None], dtype=torch.float32, device="cuda"), K[None],
                                                 size[0], size[1], sh_degree=spl["sh_degree"], render_mode="RGB")  # fmt: skip
        rgb = (img[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        frames.append(rgb)
    return _encode(frames, out)


def clean_wipe(run: Path, out: Path, *, model: int = 0, seconds: float = 7.0, size=(1280, 720), at: float = 0.35) -> Path:
    """The fused mesh (left of the wipe) against the clean model (right), shaded, from a keyframe pulled back."""
    import open3d as o3d
    import open3d.core as o3c

    from drone3d.export.render_mesh import load_parts

    model_dir, s, r, t = _frame(run, model)
    raw = o3d.io.read_triangle_mesh(str(run / "dense" / f"model_{model_dir.name}" / "mesh.ply"))
    rv = s * np.asarray(raw.vertices) @ r.T + t
    clean = load_parts(run / "export" / f"model_{model_dir.name}" / "mesh_textured.glb")
    cv_ = np.concatenate([p[0] for p in clean])
    cf = np.concatenate([p[1] + sum(len(q[0]) for q in clean[:i]) for i, p in enumerate(clean)])
    c, rot, f, w0, _ = _poses(run, model)
    _, radius = _subject(run, model)
    k = int(at * (len(c) - 1))
    fpx = f * size[0] / w0

    def shade(v: np.ndarray, faces: np.ndarray, R: np.ndarray, eye: np.ndarray) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3c.Tensor(v.astype(np.float32)), o3c.Tensor(np.asarray(faces).astype(np.uint32)))
        w, h = size
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (np.stack([(xs + 0.5 - w / 2) / fpx, (ys + 0.5 - h / 2) / fpx, np.ones_like(xs)], -1).reshape(-1, 3) @ R).astype(np.float32)
        hit = scene.cast_rays(o3c.Tensor(np.concatenate([np.broadcast_to(eye, d.shape).astype(np.float32), d], 1)))
        n = hit["primitive_normals"].numpy()
        ok = np.isfinite(hit["t_hit"].numpy())
        light = np.array([0.35, -0.45, 0.82])
        light /= np.linalg.norm(light)
        lam = 0.28 + 0.72 * np.abs(n @ light)
        img = _background(h, w).reshape(-1, 3).astype(np.float64)
        img[ok] = np.clip(lam[ok], 0, 1)[:, None] * np.array([226, 222, 214])
        return img.reshape(h, w, 3).astype(np.uint8)

    frames = []
    n = int(seconds * 30)
    for i in range(n):
        u = i / max(n - 1, 1)
        kk = min(len(c) - 1, k + int(0.15 * u * (len(c) - 1)))  # a slow drift along the flight
        R = rot[kk]
        eye = c[kk] - 0.35 * radius * R[2]
        a, b = shade(rv, np.asarray(raw.triangles), R, eye), shade(cv_, cf, R, eye)
        wipe = int(size[0] * np.clip((u - 0.2) / 0.5, 0, 1))  # the clean model sweeps in from the right
        fr = a.copy()
        fr[:, size[0] - wipe :] = b[:, size[0] - wipe :]
        if 0 < wipe < size[0]:
            fr[:, max(0, size[0] - wipe - 2) : size[0] - wipe + 2] = (245, 183, 0)
        frames.append(fr)
    return _encode(frames, out)


CLIPS = {  # segments picked from stills along each flight (promo/build/explainer/README: what each shows)
    "fly_colosseum": lambda: fly(OUT / "merge_colosseum", BUILD / "fly_colosseum.mp4", seconds=8.0, back=0.2, start=0.0, span=0.45),
    "fly_notre_dame": lambda: fly(OUT / "map_notre_dame_drone_paris_4k", BUILD / "fly_notre_dame.mp4", seconds=7.0, back=0.15,
                                  start=0.0, span=0.55),
    "fly_reichstag": lambda: fly(OUT / "map_reichstag_berlin_in_4k_stunning_drone_vi", BUILD / "fly_reichstag.mp4", seconds=6.0,
                                 back=0.1, start=0.7, span=0.3),
    "fly_rural": lambda: fly(OUT / "map_aerial_views_of_rural_riches_drone_shot", BUILD / "fly_rural.mp4", seconds=6.0, back=0.1,
                             start=0.3, span=0.7),
    "turn_complete": lambda: turntable(OUT / "new_highrise_orbit", OUT / "new_highrise_orbit" / "export" / "complete" / "scene.glb",
                                       BUILD / "turn_complete.mp4", seconds=12.0, distance=1.2, elevation=22.0, lift=0.55),
    "turn_splats": lambda: splat_turntable(OUT / "new_highrise_orbit", OUT / "new_highrise_orbit" / "splats" / "model_0_360",
                                           BUILD / "turn_splats.mp4", seconds=12.0, distance=1.2, elevation=22.0,
                                           target_z=target_z(OUT / "new_highrise_orbit",
                                                             OUT / "new_highrise_orbit" / "export" / "complete" / "scene.glb", 0.55)),
    "clean_wipe": lambda: clean_wipe(OUT / "new_highrise_orbit", BUILD / "clean_wipe.mp4"),
}


def main() -> None:
    names = sys.argv[1:] or list(CLIPS)
    for n in names:
        print(n, CLIPS[n](), flush=True)


if __name__ == "__main__":
    main()
