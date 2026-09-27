"""Typed records exchanged between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = [
    "Artifact",
    "DenseResult",
    "FrameQuality",
    "FrameRecord",
    "MeshResult",
    "PipelineResult",
    "SfMResult",
    "StageReport",
    "TelemetrySample",
    "VideoInfo",
]


@dataclass(slots=True)
class VideoInfo:
    """Basic container/codec information probed from a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str = ""

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
            "codec": self.codec,
        }
        data["megapixels"] = round(self.megapixels, 2)
        return data


@dataclass(slots=True)
class FrameQuality:
    """Per-frame quality measurements used for keyframe selection."""

    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    usable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpness": round(self.sharpness, 3),
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "overexposed_ratio": round(self.overexposed_ratio, 4),
            "underexposed_ratio": round(self.underexposed_ratio, 4),
            "usable": self.usable,
        }


@dataclass(slots=True)
class FrameRecord:
    """A frame extracted from the source video."""

    index: int
    timestamp_s: float
    path: Path
    quality: FrameQuality | None = None
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "timestamp_s": round(self.timestamp_s, 4),
            "path": str(self.path),
        }
        if self.quality is not None:
            data["quality"] = self.quality.to_dict()
        for key in ("lat", "lon", "alt_m"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(slots=True)
class TelemetrySample:
    """A single timestamped navigation sample (GPS / IMU / flight metadata)."""

    t: float
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    rel_alt_m: float | None = None
    heading_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    yaw_deg: float | None = None
    speed_ms: float | None = None
    source: str = "unknown"

    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialise as a plain dict, omitting unset (``None``) optional fields.

        Uses :func:`dataclasses.fields` rather than ``self.__dict__`` so this
        works with ``@dataclass(slots=True)``, which has no ``__dict__``.
        """
        return {
            f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None
        }


@dataclass(slots=True)
class SfMResult:
    """Outputs of a structure-from-motion stage."""

    backend: str
    model_path: Path | None = None
    sparse_ply: Path | None = None
    num_registered_images: int = 0
    num_points: int = 0
    mean_reprojection_error_px: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_path": str(self.model_path) if self.model_path else None,
            "sparse_ply": str(self.sparse_ply) if self.sparse_ply else None,
            "num_registered_images": self.num_registered_images,
            "num_points": self.num_points,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DenseResult:
    """Outputs of a dense reconstruction / depth-estimation stage."""

    backend: str
    fused_ply: Path | None = None
    depth_dir: Path | None = None
    num_points: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "fused_ply": str(self.fused_ply) if self.fused_ply else None,
            "depth_dir": str(self.depth_dir) if self.depth_dir else None,
            "num_points": self.num_points,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MeshResult:
    """Outputs of a surface reconstruction / texturing stage."""

    backend: str
    mesh_path: Path | None = None
    textured_mesh_path: Path | None = None
    num_vertices: int = 0
    num_faces: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mesh_path": str(self.mesh_path) if self.mesh_path else None,
            "textured_mesh_path": str(self.textured_mesh_path) if self.textured_mesh_path else None,
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Artifact:
    """A file produced by a stage."""

    name: str
    path: Path
    kind: str = "file"

    def to_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
        }
        try:
            if self.path.is_file():
                entry["size_bytes"] = self.path.stat().st_size
        except OSError:  # pragma: no cover - defensive
            pass
        return entry


@dataclass(slots=True)
class StageReport:
    """Execution summary for a single pipeline stage."""

    name: str
    status: str = "pending"  # ok | skipped | failed | pending
    message: str = ""
    duration_s: float = 0.0
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "duration_s": round(self.duration_s, 3),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class PipelineResult:
    """Aggregate result of a full pipeline run."""

    run_dir: Path
    stages: list[StageReport] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """No stage failed and at least one produced output.

        A run in which every stage was skipped (no video, no backends) is not
        a success: it exits non-zero instead of looking like a finished model.
        """
        return not any(s.status == "failed" for s in self.stages) and any(
            s.status == "ok" for s in self.stages
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "ok": self.ok,
            "stages": [s.to_dict() for s in self.stages],
            "metrics": self.metrics,
        }
