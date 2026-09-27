"""GPU video decoding (NVDEC via ffmpeg) straight into CUDA tensors.

The decoder runs in ffmpeg with ``-hwaccel cuda``; frames are resized on the
GPU (``scale_cuda``) and leave ffmpeg as raw NV12, 1.5 bytes a pixel. Colour
conversion to RGB happens in torch on the GPU, so the CPU only moves bytes.
Keyframes are written with nvJPEG (``torchvision.io.encode_jpeg`` on CUDA).

* :func:`stream_analysis_chunks` -- a background thread decodes every
  ``stride``-th frame at analysis size and yields RGB chunks already on the
  GPU, so decoding overlaps whatever the caller computes on each chunk.
* :func:`extract_frames` -- decodes at full (or ``long_side``) resolution and
  writes exactly the requested frame indices, numbered as the analysis pass
  numbers them.

A static ffmpeg under ``.tools/ffmpeg`` (or ``$DRONE3D_FFMPEG``) is preferred
over the system one: ffmpeg >= 5 decodes 4K VP9/AV1 on NVDEC about 1.4x faster
than the 4.4 that Ubuntu 22.04 ships, and keeps ``scale_cuda`` on the GPU.
Codecs NVDEC cannot decode fall back to multithreaded CPU decoding.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from drone3d.exceptions import IngestionError
from drone3d.logging_utils import get_logger

__all__ = [
    "StreamInfo",
    "analysis_size",
    "extract_frames",
    "ffmpeg_bin",
    "nv12_to_rgb",
    "probe_stream",
    "stream_analysis_chunks",
]

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NVDEC_CODECS = {
    "h264",
    "hevc",
    "vp8",
    "vp9",
    "av1",
    "mpeg1video",
    "mpeg2video",
    "mpeg4",
    "vc1",
    "mjpeg",
}


def ffmpeg_bin(tool: str = "ffmpeg") -> str:
    """Path of ``ffmpeg`` / ``ffprobe``: ``$DRONE3D_FFMPEG`` dir, ``.tools/ffmpeg``, then PATH."""
    import shutil

    candidates = []
    if env := os.environ.get("DRONE3D_FFMPEG"):
        candidates.append(Path(env) / tool if Path(env).is_dir() else Path(env).with_name(tool))
    candidates.append(_REPO_ROOT / ".tools" / "ffmpeg" / "bin" / tool)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    found = shutil.which(tool)
    if found is None:
        raise IngestionError(f"{tool} not found (install ffmpeg or set DRONE3D_FFMPEG)")
    return found


@dataclass(frozen=True)
class StreamInfo:
    """What ffprobe says about the first video stream."""

    path: Path
    codec: str
    width: int
    height: int
    fps: float
    num_frames: int
    duration_s: float
    color_space: str = "unknown"
    color_range: str = "tv"
    start_time_s: float = 0.0
    constant_rate: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_s": self.duration_s,
            "color_space": self.color_space,
            "color_range": self.color_range,
            "start_time_s": self.start_time_s,
            "constant_rate": self.constant_rate,
        }


def probe_stream(path: str | Path, *, count_frames: bool = True) -> StreamInfo:
    """Probe the first video stream.

    WebM/MKV carry no frame count; ``count_frames`` counts packets (one demux
    pass, no decode) so the frame numbering is exact.
    """
    video = Path(path)
    if not video.is_file():
        raise IngestionError(f"video not found: {video}")
    entries = (
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,"
        "color_space,color_range,start_time"
    )
    cmd = [ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries", entries + (",nb_read_packets" if count_frames else ""),
           "-show_entries", "format=duration", "-of", "json"]  # fmt: skip
    if count_frames:
        cmd.append("-count_packets")
    result = subprocess.run([*cmd, str(video)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise IngestionError(f"ffprobe failed on {video}: {result.stderr.strip()[-300:]}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise IngestionError(f"no video stream in {video}")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or "0/0"
    if rate in ("0/0", "0/1"):
        rate = stream.get("r_frame_rate") or "30/1"
    fps = float(Fraction(rate))
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    frames = stream.get("nb_read_packets") or stream.get("nb_frames")
    num_frames = int(frames) if frames and str(frames).isdigit() else round(duration * fps)
    return StreamInfo(
        path=video,
        codec=str(stream.get("codec_name", "")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        num_frames=num_frames,
        duration_s=duration,
        color_space=str(stream.get("color_space", "unknown")),
        color_range=str(stream.get("color_range", "tv")),
        start_time_s=float(stream.get("start_time") or 0.0),
        # avg == nominal rate means no dropped/variable intervals; timestamps
        # then map to frame numbers exactly (n = (t - start) * fps).
        constant_rate=abs(float(Fraction(stream.get("r_frame_rate") or rate)) - fps) < 1e-3 * fps,
    )


def analysis_size(width: int, height: int, long_side: int, multiple: int = 8) -> tuple[int, int]:
    """Aspect-preserving size with the given long side, both sides a multiple (RAFT needs 8)."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    scale = long_side / max(width, height)
    w = max(multiple, round(width * scale / multiple) * multiple)
    h = max(multiple, round(height * scale / multiple) * multiple)
    return int(w), int(h)


def _ffmpeg_cmd(
    info: StreamInfo, size: tuple[int, int] | None, pre_filters: list[str], hwaccel: bool
) -> list[str]:
    """ffmpeg command that writes raw NV12 frames to stdout."""
    use_gpu = hwaccel and info.codec in _NVDEC_CODECS
    if use_gpu:
        dec = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "4"]
        scale = [f"scale_cuda={size[0]}:{size[1]}:interp_algo=bilinear"] if size else []
        filters = [*pre_filters, *scale, "hwdownload", "format=nv12"]
    else:
        dec = ["-threads", "0"]
        scale = [f"scale={size[0]}:{size[1]}:flags=area"] if size else []
        filters = [*pre_filters, *scale, "format=nv12"]
    return [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", *dec,
            "-i", str(info.path), "-vf", ",".join(filters), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]  # fmt: skip


def nv12_to_rgb(nv12, height: int, width: int, *, bt709: bool = True, full_range: bool = False):  # type: ignore[no-untyped-def]
    """``uint8 [B, H*3/2, W]`` NV12 (torch, any device) -> ``uint8 [B, H, W, 3]`` RGB."""
    import torch
    import torch.nn.functional as F

    y = nv12[:, :height].float()
    uv = nv12[:, height:].reshape(-1, height // 2, width // 2, 2).permute(0, 3, 1, 2).float()
    uv = F.interpolate(uv, size=(height, width), mode="bilinear", align_corners=False)
    if full_range:
        cb, cr = uv[:, 0] - 128.0, uv[:, 1] - 128.0
    else:
        y = (y - 16.0) * (255.0 / 219.0)
        cb, cr = (uv[:, 0] - 128.0) * (255.0 / 224.0), (uv[:, 1] - 128.0) * (255.0 / 224.0)
    if bt709:
        r = y + 1.5748 * cr
        g = y - 0.187324 * cb - 0.468124 * cr
        b = y + 1.8556 * cb
    else:
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=-1).round_().clamp_(0, 255).to(torch.uint8)


def _color_args(info: StreamInfo) -> dict[str, bool]:
    space = info.color_space.lower()
    # Untagged HD/UHD video is BT.709 by convention; untagged SD is BT.601.
    bt709 = space == "bt709" or (space in ("unknown", "") and info.height >= 720)
    return {"bt709": bt709, "full_range": info.color_range.lower() in ("pc", "jpeg", "full")}


def _reader(cmd: list[str], frame_bytes: int, chunk: int, out: queue.Queue) -> None:
    """Thread body: read ``chunk``-frame blocks from ffmpeg into ``out``; ``None`` ends."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert process.stdout is not None
    try:
        while True:
            block = bytearray(frame_bytes * chunk)
            view = memoryview(block)
            filled = 0
            while filled < len(block):
                read = process.stdout.readinto(view[filled:])
                if not read:
                    break
                filled += read
            frames = filled // frame_bytes
            if frames:
                out.put(np.frombuffer(block, dtype=np.uint8, count=frames * frame_bytes))
            if filled < len(block):
                break
    finally:
        process.stdout.close()
        err = process.stderr.read().decode(errors="replace") if process.stderr else ""
        code = process.wait()
        out.put(IngestionError(f"ffmpeg exited {code}: {err.strip()[-400:]}") if code else None)


def stream_analysis_chunks(
    info: StreamInfo,
    size: tuple[int, int],
    *,
    stride: int = 1,
    chunk: int = 64,
    device: str = "cuda",
    hwaccel: bool = True,
    prefetch: int = 3,
) -> Iterator[tuple[np.ndarray, object]]:
    """Yield ``(source_frame_indices, rgb_uint8[B, H, W, 3] on device)`` chunks.

    Decoding runs ahead in a background thread (``prefetch`` chunks deep), so
    the GPU work the caller does per chunk overlaps with NVDEC.
    """
    import torch

    if stride < 1:
        raise ValueError("stride must be >= 1")
    width, height = size
    if width % 2 or height % 2:
        raise ValueError("NV12 needs even dimensions")
    pre = [f"select='not(mod(n\\,{stride}))'"] if stride > 1 else []
    cmd = _ffmpeg_cmd(info, size, pre, hwaccel)
    frame_bytes = width * height * 3 // 2
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    thread = threading.Thread(target=_reader, args=(cmd, frame_bytes, chunk, q), daemon=True)
    thread.start()
    color = _color_args(info)
    produced = 0
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            if produced == 0:
                raise item
            log.warning("%s", item)
            break
        n = len(item) // frame_bytes
        nv12 = torch.from_numpy(item).view(n, height * 3 // 2, width).to(device, non_blocking=True)
        rgb = nv12_to_rgb(nv12, height, width, **color)
        indices = (np.arange(produced, produced + n) * stride).astype(np.int64)
        produced += n
        yield indices, rgb
    thread.join()


def _decode_group(cmd: list[str], frame_bytes: int, expect: int) -> list[np.ndarray]:
    """Run one ffmpeg extraction command and return its raw NV12 frames."""
    q: queue.Queue = queue.Queue()
    _reader(cmd, frame_bytes, max(1, expect), q)
    frames: list[np.ndarray] = []
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        frames.extend(item.reshape(-1, frame_bytes))
    return frames


def extract_frames(
    info: StreamInfo,
    indices: Sequence[int],
    out_paths: Sequence[str | Path],
    *,
    long_side: int | None = None,
    quality: int = 95,
    hwaccel: bool = True,
    device: str = "cuda",
    group_size: int = 48,
    workers: int = 3,
) -> list[Path]:
    """Write source frames ``indices`` (0-based decode order) to ``out_paths``.

    The frames are split into groups of ``group_size``; each group is one
    ffmpeg process that seeks just before its first frame and selects its
    frames by timestamp, and ``workers`` groups decode concurrently on separate
    NVDEC sessions. (One select expression over hundreds of frames overflows
    ffmpeg's expression parser.) Variable-frame-rate streams, whose timestamps
    do not map to frame numbers, fall back to decoding from the start and
    selecting by frame number.

    JPEG goes through nvJPEG on ``device``; ``.png`` paths use OpenCV.
    ``long_side`` downscales on the GPU when smaller than the source.
    """
    from concurrent.futures import ThreadPoolExecutor

    import torch

    if len(indices) != len(out_paths):
        raise ValueError("indices and out_paths must have the same length")
    if not len(indices):
        return []
    order = sorted(range(len(indices)), key=lambda i: indices[i])
    wanted = [int(indices[i]) for i in order]
    if len(set(wanted)) != len(wanted):
        raise ValueError("duplicate frame indices")
    size = None
    if long_side and long_side < max(info.width, info.height):
        size = analysis_size(info.width, info.height, long_side, multiple=2)
    width, height = size or (info.width, info.height)
    frame_bytes = width * height * 3 // 2
    groups = [wanted[i : i + group_size] for i in range(0, len(wanted), group_size)]
    half = 0.5 / info.fps

    def command(group: list[int]) -> list[str]:
        if info.constant_rate:
            times = [info.start_time_s + i / info.fps for i in group]
            select = "select='" + "+".join(f"lt(abs(t-{t:.6f}),{half:.6f})" for t in times) + "'"
            cmd = _ffmpeg_cmd(info, size, [select], hwaccel)
            seek = max(0.0, times[0] - 2.0)
            at = cmd.index("-i")
            return [*cmd[:at], "-ss", f"{seek:.6f}", "-copyts", *cmd[at:]]
        select = "select='" + "+".join(f"eq(n\\,{i})" for i in group) + "'"
        return _ffmpeg_cmd(info, size, [select], hwaccel)

    try:
        from torchvision.io import encode_jpeg
    except ImportError:  # pragma: no cover - torchvision is a gpu-extra dependency
        encode_jpeg = None
    color = _color_args(info)
    written: list[Path] = [Path()] * len(indices)
    rank = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_decode_group, command(g), frame_bytes, len(g)) for g in groups]
        for group, future in zip(groups, futures, strict=True):
            frames = future.result()
            if len(frames) != len(group):
                raise IngestionError(
                    f"decoded {len(frames)} of {len(group)} frames starting at {group[0]} "
                    "(index past the end, or timestamps not constant-rate?)"
                )
            nv12 = (
                torch.from_numpy(np.stack(frames))
                .view(len(frames), height * 3 // 2, width)
                .to(device)
            )
            rgb = nv12_to_rgb(nv12, height, width, **color)
            for k in range(len(frames)):
                target = Path(out_paths[order[rank]])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.suffix.lower() in (".jpg", ".jpeg") and encode_jpeg is not None:
                    data = encode_jpeg(rgb[k].permute(2, 0, 1).contiguous(), quality=quality)
                    target.write_bytes(data.cpu().numpy().tobytes())
                else:
                    import cv2

                    cv2.imwrite(str(target), cv2.cvtColor(rgb[k].cpu().numpy(), cv2.COLOR_RGB2BGR))
                written[order[rank]] = target
                rank += 1
    return written
