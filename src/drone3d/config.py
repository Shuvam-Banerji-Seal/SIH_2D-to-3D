"""Configuration schema, YAML loading and dotted CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from drone3d.exceptions import ConfigError

__all__ = [
    "ALL_STAGES",
    "DepthConfig",
    "GeoConfig",
    "IngestConfig",
    "KeyframeConfig",
    "MeshConfig",
    "MetricsConfig",
    "PipelineConfig",
    "RenderConfig",
    "SfMConfig",
    "SplatConfig",
    "apply_override",
    "load_config",
]

ALL_STAGES: tuple[str, ...] = (
    "ingest",
    "keyframes",
    "sfm",
    "depth",
    "splat",
    "mesh",
    "georef",
    "render",
    "metrics",
    "report",
)


@dataclass
class IngestConfig:
    """Video and telemetry inputs."""

    video: str | None = None
    telemetry: str | None = None


@dataclass
class KeyframeConfig:
    """GPU pass segmentation, overlap-band keyframe selection and 3D verdict."""

    analysis_long_side: int = 640  # RAFT analysis resolution (long side, px)
    analysis_fps: float = 12.0  # frames analysed per second of video
    flow_model: str = "raft_large"  # raft_large | raft_small
    flow_batch: int = 32
    overlap_target: float = 0.75  # tau: co-visibility with the previous keyframe
    overlap_band: float = 0.10  # delta: candidates lie in [tau, tau + delta]
    max_gap_s: float = 2.0
    min_pass_s: float = 2.0
    cut_consistency: float = 0.35
    hfov_deg: float = 72.0
    parallax_snr: float = 2.0
    output_long_side: int | None = None  # None keeps the source resolution
    jpeg_quality: int = 95
    passes: list[int] = field(default_factory=list)  # restrict to these pass ids
    skip_degenerate: bool = True  # drop passes with no recoverable 3D structure
    hwaccel: bool = True


@dataclass
class SfMConfig:
    """Structure from motion (spirula-studio's GPU SfM)."""

    backend: str = "spirula"  # spirula | none
    quality: str = "high"  # low | medium | high | extreme
    camera_model: str = "radial"  # one focal: drone cameras have square pixels
    camera_mode: str = "folder"  # one camera per pass folder (edits may re-crop)
    features: str = "sift"  # sift | aliked-n16rot | aliked-n32 | loma-b128 | loma-b
    extra_args: list[str] = field(default_factory=list)


@dataclass
class DepthConfig:
    """Monocular depth prior aligned to SfM."""

    backend: str = "marigold"  # marigold | none
    checkpoint: str = "depth/Log-stage2"
    long_side: int = 1024
    batch: int = 4
    quantization: str = "4bit"  # 4bit | 8bit | none
    far_factor: float = 3.0  # beyond this x the SfM depth range counts as sky
    out_long_side: int | None = 1920


@dataclass
class SplatConfig:
    """3D Gaussian Splatting (spirula-studio trainer)."""

    backend: str = "spirula"  # spirula | none
    preset: str = "3dgs"
    iterations: int = 30000
    quality: str = "high"
    resolution_divisor: int = 2  # train on 1/divisor of the keyframe resolution
    depth_weight: float = 0.05  # Pearson depth-prior weight; 0 disables
    eval_interval: int = 8  # hold out every n-th keyframe for evaluation
    models: str = "all"  # all | largest: which SfM models to train
    min_model_images: int = 8
    flags: dict[str, object] = field(default_factory=dict)


@dataclass
class MeshConfig:
    """Mesh extraction from the trained splats."""

    backend: str = "spirula"  # spirula | none
    formats: list[str] = field(default_factory=lambda: ["ply", "glb", "obj"])
    colors: list[str] = field(default_factory=lambda: ["vertex", "texture"])


@dataclass
class GeoConfig:
    """Georeferencing from GPS priors."""

    enabled: bool = True
    origin_lat: float | None = None
    origin_lon: float | None = None
    origin_alt: float = 0.0
    align_mode: str = "auto"  # auto | similarity | yaw-scale | translation
    min_correspondences: int = 3
    write_geojson: bool = True


@dataclass
class RenderConfig:
    """Fly-through renders of the trained splats."""

    enabled: bool = True
    seconds: float = 12.0
    fps: int = 30
    long_side: int = 1920


@dataclass
class MetricsConfig:
    """Quality metrics and reporting."""

    voxel_size: float = 0.5
    reference_cloud: str | None = None
    report_thumbnails: int = 12
    gpu_telemetry: bool = True  # sample NVML utilisation/power during the run


@dataclass
class PipelineConfig:
    """Top-level configuration for a full pipeline run."""

    run_name: str = "run"
    output_root: str = "outputs"
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    log_level: str = "INFO"

    ingest: IngestConfig = field(default_factory=IngestConfig)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.stages:
            raise ConfigError("stages must not be empty")
        unknown = [stage for stage in self.stages if stage not in ALL_STAGES]
        if unknown:
            raise ConfigError(f"unknown stage(s): {', '.join(unknown)}")
        k = self.keyframes
        if not 0.0 < k.overlap_target < 1.0:
            raise ConfigError("keyframes.overlap_target must be in (0, 1)")
        if k.overlap_band <= 0 or k.overlap_target + k.overlap_band >= 1.0:
            raise ConfigError("keyframes.overlap_band must be > 0 with overlap_target + band < 1")
        if k.analysis_fps <= 0 or k.analysis_long_side < 64:
            raise ConfigError("keyframes.analysis_fps must be > 0 and analysis_long_side >= 64")
        if k.flow_model not in {"raft_large", "raft_small"}:
            raise ConfigError("keyframes.flow_model must be raft_large or raft_small")
        if self.sfm.backend not in {"spirula", "none"}:
            raise ConfigError("sfm.backend must be spirula or none")
        if self.depth.backend not in {"marigold", "none"}:
            raise ConfigError("depth.backend must be marigold or none")
        if self.splat.backend not in {"spirula", "none"}:
            raise ConfigError("splat.backend must be spirula or none")
        if self.splat.models not in {"all", "largest"}:
            raise ConfigError("splat.models must be all or largest")
        if self.mesh.backend not in {"spirula", "none"}:
            raise ConfigError("mesh.backend must be spirula or none")
        if self.geo.align_mode not in {"auto", "similarity", "yaw-scale", "translation"}:
            raise ConfigError(
                "geo.align_mode must be one of: auto, similarity, yaw-scale, translation"
            )


def _build(cls: type, data: Any, path: str = "") -> Any:
    """Recursively build a (nested) dataclass from a plain dictionary."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"'{path or cls.__name__}' must be a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{path}.{key}" if path else key
        if key not in known:
            valid = ", ".join(sorted(known))
            raise ConfigError(f"unknown configuration key '{dotted}' (valid keys: {valid})")
        hint = hints[key]
        if is_dataclass(hint):
            kwargs[key] = _build(hint, value, dotted)
        elif value is None or not isinstance(value, list | tuple):
            kwargs[key] = value
        else:
            kwargs[key] = list(value)
    return cls(**kwargs)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | tuple[str, ...] = (),
) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from YAML and apply dotted overrides.

    Args:
        path: YAML file. Missing file raises :class:`ConfigError`; ``None``
            loads built-in defaults.
        overrides: Entries of the form ``section.key=value``, e.g.
            ``preprocess.max_frames=200``. Values are parsed as YAML scalars.
    """
    data: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"configuration file not found: {config_path}")
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - passthrough
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"top level of {config_path} must be a mapping")
        data = loaded

    config = _build(PipelineConfig, data)
    for override in overrides:
        key, _, raw = override.partition("=")
        if not _:
            raise ConfigError(f"override '{override}' must use key=value syntax")
        apply_override(config, key.strip(), raw.strip())
    config.validate()
    return config


def apply_override(config: PipelineConfig, dotted: str, raw: str) -> None:
    """Set ``dotted`` (e.g. ``sfm.backend``) on ``config`` to a YAML-parsed ``raw``."""
    parts = dotted.split(".")
    target: Any = config
    for part in parts[:-1]:
        if not is_dataclass(target) or not hasattr(target, part):
            raise ConfigError(f"unknown configuration section in '{dotted}'")
        target = getattr(target, part)
    leaf = parts[-1]
    if not is_dataclass(target) or leaf not in {f.name for f in fields(target)}:
        raise ConfigError(f"unknown configuration key '{dotted}'")

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw

    current = getattr(target, leaf)
    value = _coerce(value, current, dotted)
    setattr(target, leaf, value)


def _coerce(value: Any, current: Any, dotted: str) -> Any:
    """Best-effort coercion of an override value to the field's runtime type."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false", "1", "0", "yes", "no"}:
            return value.lower() in {"true", "1", "yes"}
        raise ConfigError(f"override '{dotted}' expects a boolean")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"override '{dotted}' expects an integer") from exc
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"override '{dotted}' expects a float") from exc
    if isinstance(current, list) and not isinstance(value, list):
        if value is None:
            return []
        return [value]
    if current is None and isinstance(value, str):
        return value if value != "" else None
    return value
