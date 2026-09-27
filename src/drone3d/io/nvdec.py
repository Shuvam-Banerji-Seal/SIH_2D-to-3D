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
    "detect_letterbox",
    "extract_frames",
    "ffmpeg_bin",
    "nv12_to_rgb",
    "probe_stream",
    "sample_frames",
    "scaled_crop",
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
    info: StreamInfo,
    size: tuple[int, int] | None,
    pre_filters: list[str],
    hwaccel: bool,
    input_args: tuple[str, ...] = (),
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
    return [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", *dec, *input_args,
            "-i", str(info.path), "-vf", ",".join(filters), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]  # fmt: skip


def _ffmpeg_cmd_stacked(
    info: StreamInfo,
    size: tuple[int, int],
    side: tuple[int, int],
    pre_filters: list[str],
    hwaccel: bool,
) -> list[str]:
    """One decode, two resolutions: the ``side`` frame stacked above the ``size`` frame.

    Each output frame is NV12 of ``max(width) x (side_h + size_h)``; the analysis
    frame is padded on the right. One pipe and one reader keep the two outputs in
    lockstep with no chance of the two-pipe deadlock.
    """
    use_gpu = hwaccel and info.codec in _NVDEC_CODECS
    (w1, h1), (w2, h2) = size, side
    width = max(w1, w2)
    pre = ",".join(pre_filters)
    if use_gpu:
        dec = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "4"]
        a = f"scale_cuda={w1}:{h1}:interp_algo=bilinear,hwdownload,format=nv12"
        b = f"scale_cuda={w2}:{h2}:interp_algo=lanczos,hwdownload,format=nv12"
    else:
        dec = ["-threads", "0"]
        a, b = f"scale={w1}:{h1}:flags=area,format=nv12", f"scale={w2}:{h2}:flags=lanczos,format=nv12"
    graph = (f"[0:v]{pre + ',' if pre else ''}split=2[a][b];[a]{a},pad={width}:{h1}[aa];"
             f"[b]{b},pad={width}:{h2}[bb];[bb][aa]vstack=inputs=2[out]")  # fmt: skip
    return [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", *dec, "-i", str(info.path),
            "-filter_complex", graph, "-map", "[out]", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]  # fmt: skip


def nv12_to_rgb(
    nv12,
    height: int,
    width: int,
    *,
    bt709: bool = True,
    full_range: bool = False,
    max_batch: int = 8,
):  # type: ignore[no-untyped-def]
    """``uint8 [B, H*3/2, W]`` NV12 (torch, any device) -> ``uint8 [B, H, W, 3]`` RGB.

    Converts ``max_batch`` frames at a time: float intermediates of a 4K frame
    are ~250 MB, so a whole extraction group at once would need gigabytes.
    """
    import torch
    import torch.nn.functional as F

    out = torch.empty((nv12.shape[0], height, width, 3), dtype=torch.uint8, device=nv12.device)
    if full_range:
        ys, yo, cs = 1.0, 0.0, 1.0
    else:
        ys, yo, cs = 255.0 / 219.0, 16.0, 255.0 / 224.0
    if bt709:
        kr, kgb, kgr, kb = 1.5748, 0.187324, 0.468124, 1.8556
    else:
        kr, kgb, kgr, kb = 1.402, 0.344136, 0.714136, 1.772
    for s in range(0, nv12.shape[0], max_batch):
        chunk = nv12[s : s + max_batch]
        y = (chunk[:, :height].float() - yo) * ys
        uv = chunk[:, height:].reshape(-1, height // 2, width // 2, 2).permute(0, 3, 1, 2).float()
        uv = F.interpolate(uv, size=(height, width), mode="bilinear", align_corners=False)
        uv = (uv - 128.0) * cs
        cb, cr = uv[:, 0], uv[:, 1]
        dst = out[s : s + max_batch]
        dst[..., 0] = (y + kr * cr).round_().clamp_(0, 255)
        dst[..., 1] = (y - kgb * cb - kgr * cr).round_().clamp_(0, 255)
        dst[..., 2] = (y + kb * cb).round_().clamp_(0, 255)
    return out


def _color_args(info: StreamInfo) -> dict[str, bool]:
    space = info.color_space.lower()
    # Untagged HD/UHD video is BT.709 by convention; untagged SD is BT.601.
    bt709 = space == "bt709" or (space in ("unknown", "") and info.height >= 720)
    return {"bt709": bt709, "full_range": info.color_range.lower() in ("pc", "jpeg", "full")}


def _reader(
    cmd: list[str],
    frame_bytes: int,
    chunk: int,
    out: queue.Queue,
    stop: threading.Event | None = None,
    handle: list | None = None,
) -> None:
    """Thread body: read ``chunk``-frame blocks from ffmpeg into ``out``; ``None`` ends.

    ``stop`` lets a consumer that quits early end the thread without it
    blocking on a full queue; ``handle`` receives the ffmpeg process so the
    consumer can kill it.
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    if handle is not None:
        handle.append(process)
    assert process.stdout is not None

    def put(item: object) -> bool:
        while True:
            if stop is not None and stop.is_set():
                return False
            try:
                out.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue

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
            if frames and not put(np.frombuffer(block, dtype=np.uint8, count=frames * frame_bytes)):
                return
            if filled < len(block):
                break
    finally:
        process.stdout.close()
        err = process.stderr.read().decode(errors="replace") if process.stderr else ""
        code = process.wait()
        if not (stop is not None and stop.is_set()):
            put(IngestionError(f"ffmpeg exited {code}: {err.strip()[-400:]}") if code else None)


def stream_analysis_chunks(
    info: StreamInfo,
    size: tuple[int, int],
    *,
    stride: int = 1,
    chunk: int = 64,
    device: str = "cuda",
    hwaccel: bool = True,
    prefetch: int = 3,
    crop: tuple[int, int, int, int] | None = None,
    keyframes_only: bool = False,
    sidecar: tuple[int, int] | None = None,
) -> Iterator[tuple]:
    """Yield ``(source_frame_indices, rgb_uint8[B, H, W, 3] on device)`` chunks.

    With ``sidecar = (w, h)`` the same decode also delivers every analysis frame
    at that size and chunks are ``(indices, rgb, sidecar_rgb)`` -- keyframes can
    then be written from the first decode instead of a second 4K pass.

    Decoding runs ahead in a background thread (``prefetch`` chunks deep), so
    the GPU work the caller does per chunk overlaps with NVDEC. ``crop`` is a
    source-pixel rectangle ``(x0, y0, x1, y1)`` (e.g. from
    :func:`detect_letterbox`); it is mapped to ``size`` and rounded inwards to
    multiples of 8, as RAFT needs.
    """
    import torch

    if stride < 1:
        raise ValueError("stride must be >= 1")
    width, height = size
    if width % 2 or height % 2:
        raise ValueError("NV12 needs even dimensions")
    pre = [f"select='not(mod(n\\,{stride}))'"] if stride > 1 and not keyframes_only else []
    # keyframes_only decodes just the stream's I-frames (for sampling, not
    # analysis): indices are then ordinals, not source frame numbers.
    if sidecar is not None:
        if keyframes_only or sidecar[0] % 2 or sidecar[1] % 2:
            raise ValueError("sidecar needs even dimensions and full analysis decoding")
        cmd = _ffmpeg_cmd_stacked(info, size, sidecar, pre, hwaccel)
        stack_w, stack_h = max(width, sidecar[0]), height + sidecar[1]
        frame_bytes = stack_w * stack_h * 3 // 2
    else:
        cmd = _ffmpeg_cmd(info, size, pre, hwaccel, ("-skip_frame", "nokey") if keyframes_only else ())
        frame_bytes = width * height * 3 // 2
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop, handle = threading.Event(), []
    thread = threading.Thread(
        target=_reader, args=(cmd, frame_bytes, chunk, q, stop, handle), daemon=True
    )
    thread.start()
    color = _color_args(info)
    produced = 0
    try:
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
            if sidecar is None:
                nv12 = torch.from_numpy(item).view(n, height * 3 // 2, width).to(device, non_blocking=True)
                rgb = nv12_to_rgb(nv12, height, width, **color)
                side_rgb = None
            else:  # split the stacked planes; convert each part on its own (no chroma bleed at the seam)
                sw, sh = sidecar
                full = torch.from_numpy(item).view(n, stack_h * 3 // 2, stack_w).to(device, non_blocking=True)
                y_side, y_an = full[:, :sh, :sw], full[:, sh:stack_h, :width]
                uv_side = full[:, stack_h : stack_h + sh // 2, :sw]
                uv_an = full[:, stack_h + sh // 2 :, :width]
                rgb = nv12_to_rgb(torch.cat([y_an, uv_an], 1), height, width, **color)
                side_rgb = nv12_to_rgb(torch.cat([y_side, uv_side], 1), sh, sw, **color)
                del full
            if crop is not None:
                x0, y0, x1, y1 = scaled_crop(crop, (info.width, info.height), size, multiple=8)
                rgb = rgb[:, y0:y1, x0:x1].contiguous()
                if side_rgb is not None:
                    x0, y0, x1, y1 = scaled_crop(crop, (info.width, info.height), sidecar, multiple=2)
                    side_rgb = side_rgb[:, y0:y1, x0:x1].contiguous()
            indices = (np.arange(produced, produced + n) * stride).astype(np.int64)
            produced += n
            yield (indices, rgb) if sidecar is None else (indices, rgb, side_rgb)
    finally:
        # Reached on exhaustion and on early exit (break / close / exception):
        # never leave a decoder running behind the caller.
        stop.set()
        for process in handle:
            if process.poll() is None:
                process.kill()
        while not q.empty():
            q.get_nowait()
        thread.join(timeout=10)


def write_sidecar_jpegs(chunks: Iterator[tuple], out_dir: Path, *, quality: int = 95,
                        workers: int = 4) -> Iterator[tuple[np.ndarray, object]]:  # fmt: skip
    """Pass-through for ``stream_analysis_chunks(..., sidecar=...)``: nvJPEG-encode every
    sidecar frame to ``out_dir/f_<source index>.jpg`` and yield ``(indices, rgb)``.

    Keyframes are picked among analysis frames, so writing every candidate at the
    output size during the one decode replaces the second, seek-heavy 4K pass
    (31 s for 213 keyframes of a 55 s clip; the stacked decode is 9 s in total).
    """
    from concurrent.futures import ThreadPoolExecutor

    from torchvision.io import encode_jpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        for indices, rgb, side in chunks:
            data = encode_jpeg([f.permute(2, 0, 1).contiguous() for f in side], quality=quality)
            for idx, d in zip(indices, data, strict=True):
                pending.append(pool.submit((out_dir / f"f_{int(idx):06d}.jpg").write_bytes, d.cpu().numpy().tobytes()))
            del side
            yield indices, rgb
        for fut in pending:
            fut.result()


def scaled_crop(
    crop: tuple[int, int, int, int], src: tuple[int, int], dst: tuple[int, int], multiple: int = 2
) -> tuple[int, int, int, int]:
    """Map a source-pixel crop to a ``dst``-sized frame, rounded inwards to ``multiple``."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    x0, y0 = int(np.ceil(crop[0] * sx)), int(np.ceil(crop[1] * sy))
    x1, y1 = int(np.floor(crop[2] * sx)), int(np.floor(crop[3] * sy))
    w = (x1 - x0) // multiple * multiple
    h = (y1 - y0) // multiple * multiple
    x0 += (x1 - x0 - w) // 2
    y0 += (y1 - y0 - h) // 2
    return x0, y0, x0 + w, y0 + h


def sample_frames(
    info: StreamInfo,
    times_s: Sequence[float],
    size: tuple[int, int],
    *,
    hwaccel: bool = True,
    workers: int = 3,
) -> list[np.ndarray]:
    """Decode one frame near each timestamp (fast input seek), as NV12 arrays.

    Each sample seeks to the nearest preceding key frame and decodes one frame,
    so a dozen samples cost a few seconds however long the clip is.
    """
    from concurrent.futures import ThreadPoolExecutor

    frame_bytes = size[0] * size[1] * 3 // 2

    def one(t: float) -> np.ndarray | None:
        cmd = _ffmpeg_cmd(info, size, [], hwaccel, ("-ss", f"{max(0.0, t):.3f}"))
        at = cmd.index("-f")
        cmd = [*cmd[:at], "-frames:v", "1", *cmd[at:]]
        out = subprocess.run(cmd, capture_output=True, check=False).stdout
        return np.frombuffer(out[:frame_bytes], dtype=np.uint8) if len(out) >= frame_bytes else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [f for f in pool.map(one, times_s) if f is not None]


def detect_letterbox(
    info: StreamInfo,
    *,
    samples: int = 12,
    threshold: float = 24.0,
    hwaccel: bool = True,
) -> tuple[int, int, int, int] | None:
    """Source-pixel rectangle inside black letterbox / pillarbox bars, or ``None``.

    Bars are black in every frame, a fade only in some, so each row and column
    is judged by its *maximum* luma over ``samples`` frames spread over the
    clip. Returns ``None`` when there are no bars or when cropping would remove
    more than half the frame (a dark video, not bars).
    """
    size = analysis_size(info.width, info.height, 640, multiple=2)
    w, h = size
    duration = info.duration_s or info.num_frames / max(info.fps, 1e-9)
    times = [duration * (k + 0.5) / samples for k in range(samples)]
    frames = sample_frames(info, times, size, hwaccel=hwaccel)
    if not frames:
        return None
    luma = np.stack([f[: w * h].reshape(h, w) for f in frames]).astype(np.float32)
    if not _color_args(info)["full_range"]:
        luma = (luma - 16.0) * (255.0 / 219.0)
    peak = luma.max(axis=0)
    rows = np.flatnonzero(peak.max(axis=1) > threshold)
    cols = np.flatnonzero(peak.max(axis=0) > threshold)
    if rows.size == 0 or cols.size == 0:
        return None
    # One analysis pixel of margin past the last bar row, to skip the soft edge.
    y0 = int(rows[0]) + (1 if rows[0] > 0 else 0)
    y1 = int(rows[-1]) + (0 if rows[-1] < h - 1 else 1)
    x0 = int(cols[0]) + (1 if cols[0] > 0 else 0)
    x1 = int(cols[-1]) + (0 if cols[-1] < w - 1 else 1)
    if (y0, y1, x0, x1) == (0, h, 0, w):
        return None
    if (y1 - y0) * (x1 - x0) < 0.5 * h * w:
        log.warning("letterbox detection would keep < 50%% of the frame; not cropping")
        return None
    sx, sy = info.width / w, info.height / h
    crop = (int(round(x0 * sx)) // 2 * 2, int(round(y0 * sy)) // 2 * 2,
            int(round(x1 * sx)) // 2 * 2, int(round(y1 * sy)) // 2 * 2)  # fmt: skip
    return None if crop == (0, 0, info.width, info.height) else crop


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
    group_size: int = 24,
    workers: int = 2,
    crop: tuple[int, int, int, int] | None = None,
) -> list[Path]:
    """Write source frames ``indices`` (0-based decode order) to ``out_paths``.

    Host memory is bounded by ``workers x group_size`` raw frames (about
    12 MB each at 4K NV12: ~0.6 GB with the defaults).

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
            if crop is not None:
                x0, y0, x1, y1 = scaled_crop(
                    crop, (info.width, info.height), (width, height), multiple=2
                )
                rgb = rgb[:, y0:y1, x0:x1]
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
