"""Fly-through videos of trained splats: gsplat rasterisation + NVENC encoding.

The camera path follows the drone: it passes smoothly through the registered
SfM poses in capture order (Catmull-Rom positions, slerped orientations), so
the video shows the reconstruction from the viewpoints it was built from and
the in-between views it had to synthesise.

The splat PLY is the standard 3DGS layout spirula-studio writes (logit
opacity, log scales, wxyz quaternions, degree-3 SH in display sRGB).
``scene_transform.json`` maps the dataset frame to the frame the splats were
trained in; camera poses are carried through it.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from drone3d.exceptions import BackendUnavailable
from drone3d.logging_utils import get_logger

__all__ = ["camera_path", "load_splat_ply", "render_flythrough"]

log = get_logger(__name__)


def load_splat_ply(path: str | Path, device: str = "cuda") -> dict:
    """Read a binary little-endian 3DGS PLY into GPU tensors ready for gsplat."""
    import torch

    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: expected binary_little_endian PLY")
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    names = [line.split()[2] for line in header if line.startswith("property float")]
    data = np.frombuffer(raw, dtype="<f4", count=count * len(names), offset=end).reshape(
        count, len(names)
    )
    col = {n: i for i, n in enumerate(names)}

    def cols(prefix: str) -> np.ndarray:
        keys = sorted(
            (n for n in names if n.startswith(prefix)), key=lambda n: int(n.rsplit("_", 1)[1])
        )
        return data[:, [col[k] for k in keys]]

    rest = cols("f_rest_")  # [N, 3 * (K - 1)], channel-major
    k = rest.shape[1] // 3 + 1
    sh = np.concatenate(
        [cols("f_dc_")[:, None, :], rest.reshape(count, 3, k - 1).transpose(0, 2, 1)], axis=1
    )
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)  # noqa: E731
    return {
        "means": to(data[:, [col["x"], col["y"], col["z"]]]),
        "quats": torch.nn.functional.normalize(to(cols("rot_")), dim=-1),
        "scales": torch.exp(to(cols("scale_"))),
        "opacities": torch.sigmoid(to(data[:, col["opacity"]])),
        "sh": to(sh),
        "sh_degree": int(round(np.sqrt(k))) - 1,
    }


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(d)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _rot_to_quat(r: np.ndarray) -> np.ndarray:
    m = r
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def camera_path(
    centers: np.ndarray, rotations: np.ndarray, frames: int, smooth: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth camera path through ordered poses -> ``(centers [F, 3], cam_from_world R [F, 3, 3])``.

    Poses are box-smoothed over ``smooth`` neighbours first (SfM keyframes jitter
    slightly), then positions follow a Catmull-Rom spline and orientations slerp.
    """
    n = len(centers)
    if n < 2:
        raise ValueError("need at least two poses")
    quats = np.array([_rot_to_quat(r) for r in rotations])
    for i in range(1, n):  # keep one hemisphere so averaging and slerp are well defined
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    if smooth > 0 and n > 2 * smooth + 1:
        kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        pad = lambda a: np.pad(a, ((smooth, smooth), (0, 0)), mode="edge")  # noqa: E731
        centers = np.stack([np.convolve(c, kernel, mode="valid") for c in pad(centers).T], 1)
        quats = np.stack([np.convolve(c, kernel, mode="valid") for c in pad(quats).T], 1)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    out_c, out_r = [], []
    for f in range(frames):
        u = f / max(1, frames - 1) * (n - 1)
        i = min(int(u), n - 2)
        t = u - i
        p0, p1, p2, p3 = (
            centers[max(0, i - 1)],
            centers[i],
            centers[i + 1],
            centers[min(n - 1, i + 2)],
        )
        c = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
        )
        out_c.append(c)
        out_r.append(_quat_to_rot(_slerp(quats[i], quats[i + 1], t)))
    return np.array(out_c), np.array(out_r)


def _encoder_args() -> list[str]:
    from drone3d.io.nvdec import ffmpeg_bin

    encoders = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    if "h264_nvenc" in encoders:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "0",
        ]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


def render_flythrough(
    splat_ply: Path,
    model_dir: Path,
    out_path: Path,
    *,
    seconds: float = 12.0,
    fps: int = 30,
    long_side: int = 1920,
    batch: int = 8,
    scene_transform: Path | None = None,
    device: str = "cuda",
) -> dict:
    """Render a fly-through of ``splat_ply`` along ``model_dir``'s camera track to MP4."""
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")
    try:
        import gsplat
        import pycolmap
        import torch
    except ImportError as exc:
        raise BackendUnavailable(
            f"rendering needs gsplat and pycolmap ({exc}); uv sync --extra render"
        ) from exc
    from drone3d.io.nvdec import ffmpeg_bin

    started = time.perf_counter()
    splats = load_splat_ply(splat_ply, device)
    rec = pycolmap.Reconstruction(str(model_dir))
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    poses = [im.cam_from_world().matrix() for im in images]
    rot = np.array([p[:, :3] for p in poses])
    centers = np.array([im.projection_center() for im in images])
    s, r_tw, t_tw = 1.0, np.eye(3), np.zeros(3)
    if scene_transform and Path(scene_transform).is_file():
        tf = json.loads(Path(scene_transform).read_text())["train_from_world"]
        s, r_tw, t_tw = (
            float(tf["scale"]),
            np.array(tf["rotation"]["matrix_3x3"]),
            np.array(tf["translation"]),
        )
    frames = max(2, int(seconds * fps))
    path_c, path_r = camera_path(centers, rot, frames)
    cam = rec.cameras[images[0].camera_id]
    w0, h0 = cam.width, cam.height
    scale = long_side / max(w0, h0)
    width, height = int(round(w0 * scale / 2) * 2), int(round(h0 * scale / 2) * 2)
    params = np.asarray(cam.params, dtype=np.float64)
    model = str(cam.model).split(".")[-1]
    if model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = params[:4]
    else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL / ...: f, cx, cy, [k...]
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    # Distortion is dropped on purpose: the fly-through is rendered as a pinhole.
    k = torch.tensor(
        [[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    # cam_from_train = [R_c R_tw^T | s * t_c - R_c R_tw^T t_tw], with t_c = -R_c C (depth rescale
    # by s leaves the projection unchanged).
    views = []
    for c, r in zip(path_c, path_r, strict=True):
        rc = r @ r_tw.T
        view = np.eye(4)
        view[:3, :3] = rc
        view[:3, 3] = s * (-r @ c) - rc @ t_tw
        views.append(view)
    views_t = torch.tensor(np.array(views), dtype=torch.float32, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", *_encoder_args(),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    render_s = 0.0
    with torch.no_grad():
        for b in range(0, frames, batch):
            vm = views_t[b : b + batch]
            t0 = time.perf_counter()
            img, _, _ = gsplat.rasterization(
                splats["means"], splats["quats"], splats["scales"], splats["opacities"], splats["sh"],
                vm, k[None].expand(len(vm), 3, 3), width, height,
                sh_degree=splats["sh_degree"], render_mode="RGB", near_plane=0.01,
            )  # fmt: skip
            out = (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            render_s += time.perf_counter() - t0
            for frame in out:
                proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg encoding failed for {out_path}")
    return {
        "video": str(out_path),
        "frames": frames,
        "resolution": [width, height],
        "splats": int(splats["means"].shape[0]),
        "render_fps": round(frames / max(render_s, 1e-9), 1),
        "seconds": round(time.perf_counter() - started, 2),
    }
