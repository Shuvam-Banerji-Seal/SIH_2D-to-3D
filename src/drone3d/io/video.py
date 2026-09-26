"""Video probing and deterministic frame extraction."""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from drone3d.exceptions import IngestionError
from drone3d.logging_utils import get_logger
from drone3d.types import FrameRecord, TelemetrySample, VideoInfo

__all__ = [
    "compute_frame_step",
    "extract_frames",
    "iter_sampled_frames",
    "load_frames_csv",
    "probe_video",
    "save_frames_csv",
]

log = get_logger(__name__)

_CSV_FIELDS = (
    "index",
    "timestamp_s",
    "path",
    "lat",
    "lon",
    "alt_m",
    "sharpness",
    "brightness",
    "contrast",
    "usable",
)


def probe_video(path: str | Path) -> VideoInfo:
    """Read container metadata for ``path``.

    Raises:
        IngestionError: If the file is missing or OpenCV cannot decode it.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise IngestionError(f"video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IngestionError(f"cannot open video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise IngestionError(f"video reports invalid dimensions: {width}x{height} ({video_path})")
    fps = fps if fps > 0 else 30.0
    frame_count = max(frame_count, 0)
    codec = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)) if fourcc_value else ""
    codec = codec.replace("\x00", "").strip()
    return VideoInfo(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps if frame_count else 0.0,
        codec=codec,
    )


def compute_frame_step(native_fps: float, sample_fps: float) -> int:
    """Frames to skip between samples; ``sample_fps <= 0`` keeps every frame."""
    if sample_fps <= 0 or native_fps <= 0:
        return 1
    return max(1, int(round(native_fps / sample_fps)))


def iter_sampled_frames(
    path: str | Path,
    *,
    sample_fps: float = 0.0,
    max_frames: int | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_s, bgr_frame)`` pairs sampled at ``sample_fps``."""
    info = probe_video(path)
    step = compute_frame_step(info.fps, sample_fps)
    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():  # pragma: no cover - guarded by probe_video
        raise IngestionError(f"cannot open video: {info.path}")

    emitted = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / info.fps
            if timestamp + 1e-9 < start_s:
                index += 1
                continue
            if end_s is not None and timestamp > end_s:
                break
            if index % step == 0:
                yield timestamp, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 600,
    resize_width: int | None = None,
    image_format: str = "jpg",
    jpeg_quality: int = 92,
    telemetry: list[TelemetrySample] | None = None,
    show_progress: bool = True,
) -> list[FrameRecord]:
    """Decode, optionally subsample, and write frames as image files.

    Args:
        video: Input video path.
        out_dir: Destination directory for extracted frames.
        sample_fps: Target sampling rate (0 keeps every source frame).
        max_frames: Hard budget; candidates are uniformly decimated beyond it.
        resize_width: Optional output width in pixels (aspect ratio preserved).
        image_format: ``jpg``, ``png`` or ``webp``.
        jpeg_quality: Encoder quality for lossy formats.
        telemetry: Optional navigation samples attached to each frame.
        show_progress: Display a tqdm progress bar.

    Returns:
        Records sorted by timestamp, including GPS when telemetry is supplied.
    """
    from drone3d.io.telemetry import interpolate_sample  # local import: avoid cycle

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    info = probe_video(video)
    step = compute_frame_step(info.fps, sample_fps)

    candidates: list[int] | None = None
    if info.frame_count > 0:
        candidates = list(range(0, info.frame_count, step))
        if len(candidates) > max_frames:
            picks = np.linspace(0, len(candidates) - 1, num=max_frames).round().astype(int)
            candidates = [candidates[i] for i in sorted({int(p) for p in picks})]
            log.info("decimated %d candidate frames to %d", len(candidates), max_frames)
        wanted: set[int] | None = set(candidates)
    else:
        # Unknown-length container (frame_count <= 0): a uniform subset cannot be
        # precomputed, so honour `step` and the `max_frames` budget while streaming.
        wanted = None
        log.warning(
            "video reports no frame count; sampling every %d frame(s), budget max_frames=%d",
            step,
            max_frames,
        )

    fmt = image_format.lower().lstrip(".")
    encode_params: list[int] = []
    if fmt in {"jpg", "jpeg"}:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif fmt == "webp":
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)]
    extension = "jpg" if fmt in {"jpg", "jpeg"} else fmt

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():  # pragma: no cover - guarded by probe_video
        raise IngestionError(f"cannot open video: {info.path}")

    records: list[FrameRecord] = []
    index = 0
    progress = tqdm(
        total=len(candidates) if candidates is not None else None,
        desc="extract",
        unit="frame",
        disable=not show_progress or not sys.stderr.isatty(),
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            selected = index in wanted if wanted is not None else index % step == 0
            if selected:
                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (resize_width, int(round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                timestamp = index / info.fps
                frame_path = out_path / f"frame_{index:06d}.{extension}"
                if not cv2.imwrite(str(frame_path), frame, encode_params):
                    raise IngestionError(f"failed to write frame: {frame_path}")
                record = FrameRecord(index=index, timestamp_s=timestamp, path=frame_path)
                if telemetry:
                    sample = interpolate_sample(telemetry, timestamp)
                    if sample is not None:
                        record.lat = sample.lat
                        record.lon = sample.lon
                        record.alt_m = (
                            sample.alt_m if sample.alt_m is not None else sample.rel_alt_m
                        )
                records.append(record)
                progress.update(1)
            index += 1
            if candidates is not None and index > candidates[-1]:
                break
            if wanted is None and len(records) >= max_frames:
                log.info("hit max_frames budget (%d) on unknown-length video", max_frames)
                break
    finally:
        capture.release()
        progress.close()

    records.sort(key=lambda record: record.timestamp_s)
    log.info("extracted %d frames from %s", len(records), info.path.name)
    return records


def save_frames_csv(records: list[FrameRecord], path: str | Path) -> Path:
    """Persist frame metadata to CSV."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for record in records:
            quality = record.quality
            writer.writerow(
                {
                    "index": record.index,
                    "timestamp_s": f"{record.timestamp_s:.4f}",
                    "path": str(record.path),
                    "lat": "" if record.lat is None else f"{record.lat:.8f}",
                    "lon": "" if record.lon is None else f"{record.lon:.8f}",
                    "alt_m": "" if record.alt_m is None else f"{record.alt_m:.3f}",
                    "sharpness": "" if quality is None else f"{quality.sharpness:.3f}",
                    "brightness": "" if quality is None else f"{quality.brightness:.3f}",
                    "contrast": "" if quality is None else f"{quality.contrast:.3f}",
                    "usable": "" if quality is None else quality.usable,
                }
            )
    return csv_path


def load_frames_csv(path: str | Path) -> list[FrameRecord]:
    """Load frame metadata previously written by :func:`save_frames_csv`."""
    from drone3d.types import FrameQuality

    csv_path = Path(path)
    if not csv_path.is_file():
        raise IngestionError(f"frames manifest not found: {csv_path}")

    records: list[FrameRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            quality = None
            if row.get("sharpness"):
                quality = FrameQuality(
                    sharpness=float(row["sharpness"] or 0.0),
                    brightness=float(row.get("brightness") or 0.0),
                    contrast=float(row.get("contrast") or 0.0),
                    usable=str(row.get("usable", "True")).lower() not in {"false", "0", ""},
                )
            records.append(
                FrameRecord(
                    index=int(row["index"]),
                    timestamp_s=float(row["timestamp_s"]),
                    path=Path(row["path"]),
                    quality=quality,
                    lat=float(row["lat"]) if row.get("lat") else None,
                    lon=float(row["lon"]) if row.get("lon") else None,
                    alt_m=float(row["alt_m"]) if row.get("alt_m") else None,
                )
            )
    return records
