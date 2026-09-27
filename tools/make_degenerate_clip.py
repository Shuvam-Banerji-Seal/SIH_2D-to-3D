"""Synthesize a clip with no recoverable 3D structure, as a negative control.

A camera rotating about its own centre sees every frame as a homography of
the first, whatever the scene: ``H = K R K^-1``. Rendering such a pan (plus a
slow roll) over a single real photograph gives footage that looks like a drone
yaw sweep but carries zero parallax. The keyframe verdict must call it
degenerate.

    uv run python tools/make_degenerate_clip.py SOURCE.jpg OUT.mp4
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d.io.nvdec import ffmpeg_bin  # noqa: E402


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--yaw", type=float, default=24.0, help="total yaw sweep, degrees")
    parser.add_argument("--size", default="1920x1080")
    args = parser.parse_args()

    src = cv2.imread(args.source)
    if src is None:
        raise SystemExit(f"cannot read {args.source}")
    out_w, out_h = (int(v) for v in args.size.split("x"))
    sh, sw = src.shape[:2]
    # Source camera: 70 deg HFOV over the full photo; the output camera is a
    # narrower crop of it, so the sweep stays inside the photo.
    k_src = np.array(
        [
            [0.5 * sw / math.tan(math.radians(35)), 0, sw / 2],
            [0, 0.5 * sw / math.tan(math.radians(35)), sh / 2],
            [0, 0, 1],
        ]
    )
    f_out = 0.5 * out_w / math.tan(math.radians(22))
    k_out = np.array([[f_out, 0, out_w / 2], [0, f_out, out_h / 2], [0, 0, 1]])
    n = int(args.seconds * args.fps)
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{out_w}x{out_h}", "-r", str(args.fps), "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", args.out]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(n):
        s = i / max(1, n - 1)
        r = rotation(args.yaw * (s - 0.5), 3.0 * math.sin(2 * math.pi * s), 2.0 * (s - 0.5))
        h = k_src @ r @ np.linalg.inv(k_out)  # output pixel -> source pixel
        frame = cv2.warpPerspective(
            src, h, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {args.out}: {n} frames, pure rotation (zero parallax)")


if __name__ == "__main__":
    main()
