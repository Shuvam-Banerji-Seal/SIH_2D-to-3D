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
    "DepthConfig",
    "ExportConfig",
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
    "dense",
    "depth",
    "splat",
    "mesh",
    "georef",
    "export",
    "render",
    "metrics",
    "report",
)


@dataclass
class IngestConfig:
    """Video and telemetry inputs."""

    video: str | None = None  # the drone video (mp4, mov, mkv, avi, webm)
    telemetry: str | None = None  # flight log for georeferencing: DJI SRT, CSV, GPX or JSON (optional)
    telemetry_offset_s: float = 0.0  # the video starts this many seconds into the flight log (live segments)


@dataclass
class KeyframeConfig:
    """GPU pass segmentation, overlap-band keyframe selection and 3D verdict."""

    analysis_long_side: int = 640  # RAFT analysis resolution (long side, px)
    analysis_fps: float = 12.0  # frames analysed per second of video
    # adaptive_rate: probe the motion and analyse fewer frames when the camera moves slowly
    adaptive_rate: bool = False
    target_motion: float = 0.02  # median flow per analysis step, fraction of the width
    min_analysis_fps: float = 3.0  # lower bound for the adaptive analysis rate
    relative_overlap: bool = False  # ignore what dies in the first step (water, reflections)
    flow_model: str = "raft_large"  # raft_large | raft_small
    flow_batch: int = 32  # RAFT pairs per GPU batch (memory vs throughput)
    overlap_target: float = 0.75  # tau: co-visibility with the previous keyframe
    overlap_band: float = 0.10  # delta: candidates lie in [tau, tau + delta]
    max_gap_s: float = 2.0  # longest allowed time between keyframes; a pass is split beyond it
    min_pass_s: float = 2.0  # shorter camera moves are ignored
    cut_consistency: float = 0.35  # flow forward-backward consistency below this is a cut or fade
    hfov_deg: float = 72.0  # horizontal field of view used before SfM refines the focal length
    parallax_snr: float = 2.0  # median parallax SNR a pass needs to count as 3D (1 = no parallax)
    output_long_side: int | None = None  # None keeps the source resolution
    jpeg_quality: int = 95  # keyframe JPEG quality (nvJPEG)
    crop_letterbox: bool = True  # detect and remove black bars before analysis
    passes: list[int] = field(default_factory=list)  # restrict to these pass ids
    skip_degenerate: bool = True  # drop passes with no recoverable 3D structure
    hwaccel: bool = True  # decode with NVDEC (falls back to CPU if unavailable)


@dataclass
class SfMConfig:
    """Structure from motion (spirula-studio's GPU SfM)."""

    backend: str = "spirula"  # spirula (SIFT, GPU) | flow (RAFT tracks + pycolmap) | none
    quality: str = "high"  # low | medium | high | extreme
    camera_model: str = "radial"  # one focal: drone cameras have square pixels
    camera_mode: str = "folder"  # one camera per pass folder (edits may re-crop)
    features: str = "sift"  # sift | aliked-n16rot | aliked-n32 | loma-b128 | loma-b
    extra_args: list[str] = field(default_factory=list)  # extra command-line arguments for spirula sfm
    # backend: flow
    flow_long_side: int = 960  # tracking resolution; keypoints are written at full resolution
    flow_span: int = 3  # direct flow to the next N keyframes
    flow_stride: int = 16  # seeding grid (px at the tracking resolution)
    flow_max_gap: int = 4  # keyframe pairs matched per track
    mapper: str = "global"  # global (GLOMAP; registered every pass of our footage) | incremental


@dataclass
class DenseConfig:
    """Dense depth from flow triangulation (+ monocular fill) fused in a GPU TSDF."""

    backend: str = "flow"  # flow | none
    long_side: int = 480  # depth-map resolution (px); 960 is sharper and ~4x slower than 480
    gaps: list[int] = field(default_factory=lambda: [2, 4, 8, 12])  # in keyframes
    keyframe_stride: int = 1  # depth maps for every N-th keyframe (the TSDF still fuses them all)
    min_angle_deg: float = 0.5  # smallest triangulation angle kept (single-pass keyframes are 0.1-1 deg apart)
    rel_tol: float = 0.05  # neighbour depths must agree within this fraction to be fused
    mono_model: str | None = "depth-anything/Depth-Anything-V2-Large-hf"  # null: triangulated depth only
    # TSDF: the truncation band drives completeness. Jal Mahal's two largest models, band 4 -> 12
    # voxels: completeness 0.47 -> 0.80 and 0.75 -> 0.91 (depth maps cover all non-sky pixels, but
    # neighbouring views disagree slightly and a narrow band lets them cancel), median depth error
    # against the triangulated depth 0.42 -> 0.60 % and 0.92 -> 1.09 %. 3 px voxels match 2 px with
    # fewer triangles.
    voxel_px: float = 3.0  # TSDF voxel in pixel footprints at the median depth
    trunc_voxels: float = 12.0  # TSDF truncation band in voxels (4 -> 12: completeness 0.47 -> 0.80)
    tsdf_memory_gb: float = 8.0  # GPU memory for the TSDF hash map; the voxel grows if the scene needs more
    # auto | always | never: fuse in a child process (auto: inside the warm engine, where Open3D's CUDA state
    # accumulates over many videos; a fault then kills the child, not the engine)
    isolate_fusion: str = "auto"
    min_model_images: int = 3  # SfM models with fewer registered keyframes are skipped


@dataclass
class ExportConfig:
    """Deliverable formats and the web viewer."""

    enabled: bool = True  # write the deliverables and the viewer
    mesh_formats: list[str] = field(default_factory=lambda: ["ply", "obj", "glb", "fbx", "stl", "blend"])  # ply | obj | glb | fbx | stl | blend
    las: bool = True  # write the point cloud as LAS (UTM + EPSG when georeferenced)
    geotiff: bool = True  # DSM + orthophoto (projected UTM when georeferenced)
    raster_cell: float | None = None  # metres (or model units); default: 2 x median point spacing
    viewer: bool = True  # write the three.js web viewer
    texture: bool = True  # bake an atlas from the keyframes (sharper than TSDF vertex colours)
    texture_views: int = 16  # candidate keyframes; each triangle takes its best view
    texture_size: int = 4096  # texture atlas size (px)
    max_triangles: int = 600_000  # viewable copies (GLB, textured, FBX); mesh.ply keeps full density
    splats: bool = True  # convert trained Gaussian splats (splat stage) for the web viewer
    max_splats: int = 1_500_000  # the most important splats kept in the web file (32 bytes each)


@dataclass
class DepthConfig:
    """Monocular depth prior aligned to SfM."""

    backend: str = "marigold"  # marigold | none
    checkpoint: str = "depth/Log-stage2"  # Marigold v2 checkpoint (depth/Log-stage2 or normals)
    long_side: int = 1024  # prediction resolution (px)
    batch: int = 4  # keyframes per Marigold batch
    quantization: str = "4bit"  # 4bit | 8bit | none
    far_factor: float = 3.0  # beyond this x the SfM depth range counts as sky
    out_long_side: int | None = 1920  # written depth-map resolution (px)
    calibration: str = "monotone"  # monotone (isotonic vs SfM) | affine


@dataclass
class SplatConfig:
    """3D Gaussian Splatting (spirula-studio trainer)."""

    backend: str = "spirula"  # spirula | none
    preset: str = "3dgs"  # spirula-studio training preset
    iterations: int = 30000  # training steps per model
    quality: str = "high"  # low | medium | high | extreme
    resolution_divisor: int = 2  # train on 1/divisor of the keyframe resolution
    depth_weight: float = 0.05  # Pearson depth-prior weight; 0 disables
    eval_interval: int = 8  # hold out every n-th keyframe for evaluation
    models: str = "all"  # all | largest: which SfM models to train
    init: str = "dense"  # dense | sparse: start from the dense TSDF cloud (when the dense stage ran) or the SfM points
    init_points: int = 400_000  # dense starting points per model (random subset)
    min_model_images: int = 8  # smaller SfM models are not trained
    cache_images: str = "disk"  # disk | cpu: cpu is faster but holds every decoded image in RAM
    flags: dict[str, object] = field(default_factory=dict)  # extra spirula train flags (key: value)


@dataclass
class MeshConfig:
    """Mesh extraction from the trained splats."""

    backend: str = "spirula"  # spirula | none
    formats: list[str] = field(default_factory=lambda: ["ply", "glb", "obj"])  # mesh files written from the splats
    colors: list[str] = field(default_factory=lambda: ["vertex", "texture"])  # vertex colours and/or texture atlas
    num_threads: int = 12  # CPU threads for meshing (0 = all; the host may be shared)


@dataclass
class GeoConfig:
    """Georeferencing from GPS priors."""

    enabled: bool = True  # georeference when the keyframes carry GPS
    origin_lat: float | None = None  # ENU origin latitude (default: median of the GPS fixes)
    origin_lon: float | None = None  # ENU origin longitude (default: median of the GPS fixes)
    origin_alt: float = 0.0  # ENU origin altitude when the log has none (m)
    align_mode: str = "auto"  # auto | similarity | yaw-scale | translation
    min_correspondences: int = 3  # GPS-tagged keyframes a model needs to be georeferenced
    write_geojson: bool = True  # write the camera track as GeoJSON


@dataclass
class RenderConfig:
    """Fly-through renders of the trained splats."""

    enabled: bool = True  # render fly-through videos of the splats
    seconds: float = 12.0  # fly-through length
    fps: int = 30  # frames per second
    long_side: int = 1920  # video resolution (px)


@dataclass
class MetricsConfig:
    """Quality metrics and reporting."""

    voxel_size: float = 0.5  # voxel size for coverage statistics (m when georeferenced, else model units)
    # optional ground-truth cloud: LAS/LAZ in a projected CRS (georeferenced runs), or PLY/PCD/XYZ
    # in the export frame; gives accuracy and completeness of every model against it
    reference_cloud: str | None = None
    reference_threshold_m: float = 1.0  # a reference point counts as reconstructed within this distance (PS: <= 1 m)
    report_thumbnails: int = 12  # keyframes shown in the HTML report
    gpu_telemetry: bool = True  # sample NVML utilisation/power during the run


@dataclass
class PipelineConfig:
    """Top-level configuration for a full pipeline run."""

    run_name: str = "run"  # run directory name under output_root
    output_root: str = "outputs"  # where run directories are created
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))  # stages to run, in pipeline order
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR

    ingest: IngestConfig = field(default_factory=IngestConfig)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
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
        if self.sfm.backend not in {"spirula", "flow", "none"}:
            raise ConfigError("sfm.backend must be spirula, flow or none")
        if self.sfm.mapper not in {"incremental", "global"}:
            raise ConfigError("sfm.mapper must be incremental or global")
        if self.dense.backend not in {"flow", "none"}:
            raise ConfigError("dense.backend must be flow or none")
        if self.dense.isolate_fusion not in {"auto", "always", "never"}:
            raise ConfigError("dense.isolate_fusion must be auto, always or never")
        if not self.dense.gaps or min(self.dense.gaps) < 1:
            raise ConfigError("dense.gaps must be positive keyframe offsets")
        if self.depth.backend not in {"marigold", "none"}:
            raise ConfigError("depth.backend must be marigold or none")
        if self.splat.backend not in {"spirula", "none"}:
            raise ConfigError("splat.backend must be spirula or none")
        if self.splat.init not in {"dense", "sparse"}:
            raise ConfigError("splat.init must be dense or sparse")
        if self.splat.models not in {"all", "largest"}:
            raise ConfigError("splat.models must be all or largest")
        if self.mesh.backend not in {"spirula", "none"}:
            raise ConfigError("mesh.backend must be spirula or none")
        if self.metrics.voxel_size <= 0 or self.metrics.reference_threshold_m <= 0:
            raise ConfigError("metrics.voxel_size and metrics.reference_threshold_m must be > 0")
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
