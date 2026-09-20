"""Configuration schema, YAML loading and dotted CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from drone3d.exceptions import ConfigError

__all__ = [
    "ALL_STAGES",
    "DenseConfig",
    "GeoConfig",
    "IngestConfig",
    "MeshConfig",
    "MetricsConfig",
    "PipelineConfig",
    "PreprocessConfig",
    "SfMConfig",
    "apply_override",
    "load_config",
]

ALL_STAGES: tuple[str, ...] = (
    "ingest",
    "preprocess",
    "sfm",
    "dense",
    "mesh",
    "georef",
    "metrics",
    "report",
)


@dataclass
class IngestConfig:
    """Video and telemetry inputs."""

    video: str | None = None
    telemetry: str | None = None
    frames_dir: str = "frames"
    image_format: str = "jpg"
    jpeg_quality: int = 92


@dataclass
class PreprocessConfig:
    """Frame sampling, quality filtering and image restoration."""

    sample_fps: float = 2.0
    max_frames: int = 600
    oversample: int = 3
    min_sharpness: float = 20.0
    min_spacing_s: float = 0.2
    resize_width: int | None = None
    selection: str = "sharpness"  # sharpness | uniform
    deblur: bool = False
    deblur_sigma: float = 1.4
    deblur_amount: float = 0.8
    stabilize: bool = False
    dynamic_masking: bool = False


@dataclass
class SfMConfig:
    """Structure-from-motion backend settings."""

    backend: str = "colmap"  # colmap | none
    binary: str = "colmap"
    matcher: str = "sequential"  # sequential | exhaustive | vocab_tree
    camera_model: str = "OPENCV"
    single_camera: bool = True
    max_features: int = 8192
    use_gps_priors: bool = True
    allow_missing: bool = True
    extra_args: list[str] = field(default_factory=list)


@dataclass
class DenseConfig:
    """Dense reconstruction / depth estimation settings."""

    backend: str = "mvs"  # mvs | mono | none
    max_image_size: int = 2000
    geom_consistency: bool = True
    mono_depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    allow_missing: bool = True


@dataclass
class MeshConfig:
    """Surface extraction and texturing settings."""

    backend: str = "poisson"  # poisson | delaunay | none
    depth: int = 11
    trim: float = 10.0
    texture: bool = True
    target_faces: int | None = None
    allow_missing: bool = True


@dataclass
class GeoConfig:
    """Georeferencing from GPS/IMU priors."""

    enabled: bool = True
    origin_lat: float | None = None
    origin_lon: float | None = None
    origin_alt: float = 0.0
    epsg: int | None = 4326
    align_mode: str = "similarity"  # similarity | translation | none
    min_correspondences: int = 3
    write_geojson: bool = True


@dataclass
class MetricsConfig:
    """Quality metrics and reporting."""

    voxel_size: float = 0.5
    reference_cloud: str | None = None
    expected_extent_m: float | None = None
    report_thumbnails: int = 12
    make_contact_sheet: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for a full pipeline run."""

    run_name: str = "run"
    output_root: str = "outputs"
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    log_level: str = "INFO"
    seed: int = 0

    ingest: IngestConfig = field(default_factory=IngestConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        unknown = [stage for stage in self.stages if stage not in ALL_STAGES]
        if unknown:
            raise ConfigError(f"unknown stage(s): {', '.join(unknown)}")
        if self.preprocess.sample_fps < 0:
            raise ConfigError("preprocess.sample_fps must be >= 0 (0 = every frame)")
        if self.preprocess.max_frames < 1:
            raise ConfigError("preprocess.max_frames must be >= 1")
        if self.preprocess.oversample < 1:
            raise ConfigError("preprocess.oversample must be >= 1")
        if self.geo.align_mode not in {"similarity", "translation", "none"}:
            raise ConfigError("geo.align_mode must be one of: similarity, translation, none")
        if self.sfm.matcher not in {"sequential", "exhaustive", "vocab_tree"}:
            raise ConfigError("sfm.matcher must be one of: sequential, exhaustive, vocab_tree")


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
