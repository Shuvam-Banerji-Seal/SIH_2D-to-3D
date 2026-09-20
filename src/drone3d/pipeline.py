"""End-to-end orchestration of the single-pass drone reconstruction pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drone3d.config import ALL_STAGES, PipelineConfig
from drone3d.exceptions import BackendUnavailable, Drone3DError
from drone3d.geo.georef import gps_rmse, solve_similarity
from drone3d.geo.projection import LocalTangentPlane
from drone3d.io.telemetry import load_telemetry, telemetry_summary
from drone3d.io.video import extract_frames, load_frames_csv, probe_video, save_frames_csv
from drone3d.logging_utils import get_logger
from drone3d.metrics.quality import cloud_bounds, geometric_scale_error, summarize_cloud
from drone3d.preprocess.deblur import auto_deblur
from drone3d.preprocess.dynamic import DynamicMasker
from drone3d.preprocess.quality import measure_frames, select_frames
from drone3d.preprocess.stabilize import stabilize_frames
from drone3d.report.html import generate_report, make_contact_sheet
from drone3d.sfm.base import get_sfm_backend
from drone3d.types import Artifact, PipelineResult, SfMResult, StageReport
from drone3d.utils.ply import load_ply, write_ply
from drone3d.version import __version__

__all__ = ["Pipeline", "default_run_dir"]

log = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def default_run_dir(config: PipelineConfig) -> Path:
    """Timestamped run directory under ``config.output_root``."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(config.output_root) / f"{config.run_name}_{timestamp}"


def _write_json(path: str | Path, payload: Any) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return json_path


def _read_json(path: str | Path) -> dict[str, Any] | None:
    json_path = Path(path)
    if not json_path.is_file():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("ignoring corrupt JSON artifact: %s", json_path)
        return None


class Pipeline:
    """Runs pipeline stages against a single run directory.

    Stages communicate exclusively through artifacts (JSON/CSV/PLY) inside the
    run directory, so any stage can be re-executed on its own, e.g.
    ``drone3d run --stages mesh`` after tuning mesh settings.
    """

    def __init__(
        self,
        config: PipelineConfig,
        run_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir) if run_dir is not None else default_run_dir(config)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, Any] = {}
        self._result = PipelineResult(run_dir=self.run_dir)

    # ------------------------------------------------------------------ public

    def run(self, stages: list[str] | None = None) -> PipelineResult:
        """Execute ``stages`` (default: configured stages) in order."""
        selected = list(stages or self.config.stages)
        unknown = [stage for stage in selected if stage not in ALL_STAGES]
        if unknown:
            raise Drone3DError(f"unknown stage(s): {', '.join(unknown)}")

        self._result = PipelineResult(run_dir=self.run_dir)
        for name in selected:
            self._result.stages.append(self._run_stage(name))

        self._metrics = self._collect_metrics()
        self._result.metrics = self._metrics
        manifest_path = self._write_manifest()
        log.info("manifest written: %s", manifest_path)
        return self._result

    # ----------------------------------------------------------- stage plumbing

    def _run_stage(self, name: str) -> StageReport:
        handler = getattr(self, f"_stage_{name}", None)
        if handler is None:
            return StageReport(name, "failed", f"no handler for stage '{name}'")
        log.info("=== stage: %s ===", name)
        started = time.perf_counter()
        try:
            report: StageReport = handler()
        except BackendUnavailable as exc:
            report = StageReport(name, "skipped", str(exc))
        except Drone3DError as exc:
            report = StageReport(name, "failed", str(exc))
            log.error("stage %s failed: %s", name, exc)
        except Exception as exc:  # keep the run alive to finish other stages
            report = StageReport(name, "failed", f"{type(exc).__name__}: {exc}")
            log.exception("stage %s crashed", name)
        report.duration_s = time.perf_counter() - started
        log.info("stage %s: %s (%.2fs) %s", name, report.status, report.duration_s, report.message)
        return report

    def _stage_dir(self, stage: str) -> Path:
        path = self.run_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _stage_result(self, stage: str) -> dict[str, Any] | None:
        return _read_json(self.run_dir / stage / "result.json")

    def _images_dir(self) -> Path | None:
        """Prefer selected frames; fall back to raw extracted frames."""
        for candidate in (
            self.run_dir / "frames_selected",
            self.run_dir / self.config.ingest.frames_dir,
        ):
            if candidate.is_dir() and any(
                path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir()
            ):
                return candidate
        return None

    # ------------------------------------------------------------------- stages

    def _stage_ingest(self) -> StageReport:
        cfg = self.config.ingest
        pre = self.config.preprocess
        if not cfg.video:
            return StageReport("ingest", "skipped", "ingest.video is not configured")

        stage_dir = self._stage_dir("ingest")
        telemetry = load_telemetry(cfg.telemetry) if cfg.telemetry else []
        if cfg.telemetry:
            log.info("loaded %d telemetry samples from %s", len(telemetry), cfg.telemetry)

        info = probe_video(cfg.video)
        frames_dir = self.run_dir / cfg.frames_dir
        records = extract_frames(
            cfg.video,
            frames_dir,
            sample_fps=pre.sample_fps,
            max_frames=max(1, pre.max_frames * pre.oversample),
            resize_width=pre.resize_width,
            image_format=cfg.image_format,
            jpeg_quality=cfg.jpeg_quality,
            telemetry=telemetry or None,
        )
        frames_csv = save_frames_csv(records, stage_dir / "frames.csv")
        info_path = _write_json(stage_dir / "video_info.json", info.to_dict())
        telemetry_stats = telemetry_summary(telemetry)
        telemetry_path = _write_json(stage_dir / "telemetry_summary.json", telemetry_stats)

        _write_json(
            stage_dir / "result.json",
            {
                "video": info.to_dict(),
                "num_frames": len(records),
                "frames_csv": str(frames_csv),
                "telemetry": telemetry_stats,
            },
        )
        artifacts = [
            Artifact("frames", frames_dir, "directory"),
            Artifact("frames_csv", frames_csv),
            Artifact("video_info", info_path),
            Artifact("telemetry_summary", telemetry_path),
        ]
        return StageReport(
            "ingest",
            "ok",
            f"{len(records)} frames sampled at {pre.sample_fps} fps from {info.duration_s:.1f}s",
            artifacts=artifacts,
            metrics={
                "num_frames": len(records),
                "video": info.to_dict(),
                "telemetry": telemetry_stats,
            },
        )

    def _stage_preprocess(self) -> StageReport:
        cfg = self.config.preprocess
        frames_csv = self.run_dir / "ingest" / "frames.csv"
        if not frames_csv.is_file():
            return StageReport("preprocess", "skipped", "no ingest output (run ingest first)")
        records = load_frames_csv(frames_csv)
        if not records:
            return StageReport("preprocess", "skipped", "ingest produced zero frames")

        measure_frames(records, blur_threshold=cfg.min_sharpness)
        selected = select_frames(
            records,
            max_frames=cfg.max_frames,
            min_sharpness=cfg.min_sharpness,
            min_spacing_s=cfg.min_spacing_s,
            selection=cfg.selection,
        )
        if not selected:
            return StageReport("preprocess", "skipped", "no frame passed quality filtering")

        selected_dir = self.run_dir / "frames_selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = selected_dir / "masks"
        pairs: list[tuple[Any, np.ndarray]] = []
        for record in selected:
            image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
            if image is None:
                log.warning("skipping unreadable frame: %s", record.path)
                continue
            pairs.append((record, image))

        if cfg.stabilize and len(pairs) > 1:
            stabilized = stabilize_frames([image for _, image in pairs])
            pairs = [(record, image) for (record, _), image in zip(pairs, stabilized, strict=True)]

        masker = DynamicMasker() if cfg.dynamic_masking else None
        mask_ratios: list[float] = []
        saved = 0
        for record, image in pairs:
            if masker is not None:
                mask = masker.mask(image)
                mask_ratios.append(float(np.count_nonzero(mask)) / float(mask.size))
                mask_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_dir / f"{record.path.stem}.png"), mask)
            if cfg.deblur:
                image = auto_deblur(
                    image,
                    threshold=cfg.min_sharpness * 3.0,
                    sigma=cfg.deblur_sigma,
                    amount=cfg.deblur_amount,
                )
            target = selected_dir / record.path.name
            if not cv2.imwrite(str(target), image):
                log.warning("failed to write frame: %s", target)
                continue
            record.path = target
            saved += 1

        if saved == 0:
            return StageReport("preprocess", "failed", "no frame could be written")

        selected_csv = save_frames_csv(
            selected, self._stage_dir("preprocess") / "selected_frames.csv"
        )
        sharpness = [record.quality.sharpness for record in selected if record.quality]
        summary = {
            "candidates": len(records),
            "selected": saved,
            "max_frames": cfg.max_frames,
            "selection": cfg.selection,
            "deblur": cfg.deblur,
            "stabilize": cfg.stabilize,
            "dynamic_masking": cfg.dynamic_masking,
            "sharpness": {
                "min": round(min(sharpness), 3) if sharpness else None,
                "mean": round(float(np.mean(sharpness)), 3) if sharpness else None,
                "max": round(max(sharpness), 3) if sharpness else None,
            },
            "dynamic_mask_ratio_mean": round(float(np.mean(mask_ratios)), 4)
            if mask_ratios
            else None,
        }
        summary_path = _write_json(
            self._stage_dir("preprocess") / "selection_summary.json", summary
        )
        _write_json(self._stage_dir("preprocess") / "result.json", summary)
        artifacts = [
            Artifact("selected_frames", selected_dir, "directory"),
            Artifact("selected_frames_csv", selected_csv),
            Artifact("selection_summary", summary_path),
        ]
        return StageReport(
            "preprocess",
            "ok",
            f"selected {saved}/{len(records)} frames",
            artifacts=artifacts,
            metrics=summary,
        )

    def _stage_sfm(self) -> StageReport:
        cfg = self.config.sfm
        images_dir = self._images_dir()
        if images_dir is None:
            return StageReport(
                "sfm", "skipped", "no frames available (run ingest/preprocess first)"
            )

        backend = get_sfm_backend(cfg.backend, cfg.binary)
        if not backend.is_available() and cfg.allow_missing:
            return StageReport(
                "sfm",
                "skipped",
                f"SfM backend '{backend.name}' unavailable; install COLMAP or set sfm.backend=none",
            )
        stage_dir = self._stage_dir("sfm")
        result: SfMResult = backend.reconstruct(images_dir, stage_dir, cfg)
        result_path = _write_json(stage_dir / "result.json", result.to_dict())
        artifacts = [Artifact("sfm_result", result_path)]
        if result.sparse_ply is not None:
            artifacts.append(Artifact("sparse_cloud", result.sparse_ply))
        return StageReport(
            "sfm",
            "ok",
            f"registered {result.num_registered_images} images, {result.num_points} points",
            artifacts=artifacts,
            metrics=result.to_dict(),
        )

    def _load_sfm_result(self) -> SfMResult | None:
        payload = self._stage_result("sfm")
        if payload is None:
            return None
        return SfMResult(
            backend=payload.get("backend", "unknown"),
            model_path=Path(payload["model_path"]) if payload.get("model_path") else None,
            sparse_ply=Path(payload["sparse_ply"]) if payload.get("sparse_ply") else None,
            num_registered_images=int(payload.get("num_registered_images", 0)),
            num_points=int(payload.get("num_points", 0)),
            mean_reprojection_error_px=payload.get("mean_reprojection_error_px"),
            metadata=payload.get("metadata", {}),
        )

    def _stage_dense(self) -> StageReport:
        cfg = self.config.dense
        sfm = self._load_sfm_result()
        if sfm is None:
            return StageReport("dense", "skipped", "no SfM result (run sfm first)")
        images_dir = self._images_dir()
        if images_dir is None:
            return StageReport("dense", "skipped", "no images available")

        from drone3d.dense.base import get_dense_backend

        backend = get_dense_backend(cfg.backend, binary=self.config.sfm.binary)
        if not backend.is_available() and cfg.allow_missing:
            return StageReport(
                "dense",
                "skipped",
                f"dense backend '{backend.name}' unavailable; install COLMAP or uv sync --extra ai",
            )
        stage_dir = self._stage_dir("dense")
        result = backend.reconstruct(images_dir, sfm, stage_dir, cfg)
        result_path = _write_json(stage_dir / "result.json", result.to_dict())
        artifacts = [Artifact("dense_result", result_path)]
        if result.fused_ply is not None:
            artifacts.append(Artifact("dense_cloud", result.fused_ply))
        return StageReport(
            "dense",
            "ok",
            f"{result.num_points} dense points" if result.fused_ply else "depth maps written",
            artifacts=artifacts,
            metrics=result.to_dict(),
        )

    def _stage_mesh(self) -> StageReport:
        cfg = self.config.mesh
        dense = self._stage_result("dense")
        if dense is None:
            return StageReport("mesh", "skipped", "no dense result (run dense first)")
        fused_ply = dense.get("fused_ply")
        if not fused_ply:
            return StageReport(
                "mesh",
                "skipped",
                "mesh extraction needs a fused cloud (set dense.backend=mvs)",
            )
        images_dir = self._images_dir()
        if images_dir is None:
            return StageReport("mesh", "skipped", "no images available for texturing")

        from drone3d.mesh.base import get_mesh_backend

        backend = get_mesh_backend(cfg.backend, binary=self.config.sfm.binary)
        if not backend.is_available() and cfg.allow_missing:
            return StageReport("mesh", "skipped", f"mesh backend '{backend.name}' unavailable")
        stage_dir = self._stage_dir("mesh")
        result = backend.build(Path(fused_ply), images_dir, stage_dir, cfg)
        result_path = _write_json(stage_dir / "result.json", result.to_dict())
        artifacts = [Artifact("mesh_result", result_path)]
        if result.mesh_path is not None:
            artifacts.append(Artifact("mesh", result.mesh_path))
        if result.textured_mesh_path is not None:
            artifacts.append(Artifact("textured_mesh", result.textured_mesh_path))
        return StageReport(
            "mesh",
            "ok",
            f"{result.num_vertices} vertices / {result.num_faces} faces",
            artifacts=artifacts,
            metrics=result.to_dict(),
        )

    def _stage_georef(self) -> StageReport:
        cfg = self.config.geo
        if not cfg.enabled:
            return StageReport("georef", "skipped", "geo.enabled is false")

        sfm = self._load_sfm_result()
        if sfm is None:
            return StageReport("georef", "skipped", "no SfM result (run sfm first)")
        text_model = Path(sfm.metadata.get("text_model", "")) if sfm.metadata else None
        images_txt = text_model / "images.txt" if text_model else None
        if images_txt is None or not images_txt.is_file():
            return StageReport("georef", "skipped", "COLMAP text model not available")

        frames = load_frames_csv(self.run_dir / "ingest" / "frames.csv")
        frame_lookup = {
            Path(record.path).name: record
            for record in frames
            if record.lat is not None and record.lon is not None
        }
        if not frame_lookup:
            return StageReport(
                "georef", "skipped", "no frame GPS fixes (ingest.telemetry missing?)"
            )

        from drone3d.sfm.colmap_model import parse_images_text

        poses = parse_images_text(images_txt)
        matches = [(pose, frame_lookup[pose.name]) for pose in poses if pose.name in frame_lookup]
        if len(matches) < cfg.min_correspondences:
            return StageReport(
                "georef",
                "skipped",
                f"only {len(matches)} GPS/image correspondences "
                f"(need >= {cfg.min_correspondences})",
            )

        lats = [record.lat for _, record in matches if record.lat is not None]
        lons = [record.lon for _, record in matches if record.lon is not None]
        alts = [record.alt_m for _, record in matches if record.alt_m is not None]
        origin_lat = cfg.origin_lat if cfg.origin_lat is not None else float(np.median(lats))
        origin_lon = cfg.origin_lon if cfg.origin_lon is not None else float(np.median(lons))
        origin_alt = float(np.median(alts)) if alts else cfg.origin_alt
        plane = LocalTangentPlane(origin_lat, origin_lon, origin_alt)

        model_centers = np.array([pose.center for pose, _ in matches], dtype=np.float64)
        gps_local = np.array(
            [
                plane.to_local(record.lat, record.lon, record.alt_m or origin_alt)
                for _, record in matches
            ],
            dtype=np.float64,
        )

        if cfg.align_mode == "translation":
            from drone3d.geo.georef import SimilarityTransform

            transform = SimilarityTransform(
                scale=1.0,
                rotation=np.eye(3),
                translation=gps_local.mean(axis=0) - model_centers.mean(axis=0),
            )
        else:
            transform = solve_similarity(
                model_centers,
                gps_local,
                with_scale=cfg.align_mode == "similarity",
            )
        accuracy = gps_rmse(transform.apply(model_centers), gps_local)

        stage_dir = self._stage_dir("georef")
        artifacts: list[Artifact] = []
        georeferenced: list[Path] = []
        dense_payload = self._stage_result("dense") or {}
        clouds_to_transform: list[tuple[str, Path | None]] = [
            ("sparse", sfm.sparse_ply),
            ("dense", Path(dense_payload["fused_ply"]) if dense_payload.get("fused_ply") else None),
        ]
        for label, cloud_path in clouds_to_transform:
            if cloud_path is None or not Path(cloud_path).is_file():
                continue
            cloud = load_ply(cloud_path)
            transformed = transform.apply(cloud.points)
            output = stage_dir / f"georeferenced_{label}.ply"
            write_ply(output, transformed, cloud.colors)
            georeferenced.append(output)
            artifacts.append(Artifact(f"georeferenced_{label}", output))

        geojson_path = None
        if cfg.write_geojson:
            geojson_path = _write_json(
                stage_dir / "camera_track.geojson",
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [record.lon, record.lat]},
                            "properties": {"name": pose.name, "alt_m": record.alt_m},
                        }
                        for pose, record in matches
                    ],
                },
            )
            artifacts.append(Artifact("camera_track", geojson_path))

        payload = {
            "origin": plane.to_dict(),
            "align_mode": cfg.align_mode,
            "epsg": cfg.epsg,
            "num_correspondences": len(matches),
            "transform": transform.to_dict(),
            "gps_accuracy": accuracy,
            "georeferenced_clouds": [str(path) for path in georeferenced],
            "camera_track": str(geojson_path) if geojson_path else None,
        }
        result_path = _write_json(stage_dir / "result.json", payload)
        artifacts.append(Artifact("georef_result", result_path))
        return StageReport(
            "georef",
            "ok",
            f"{len(matches)} tie points, horizontal RMSE {accuracy['rmse_horizontal_m']:.2f} m",
            artifacts=artifacts,
            metrics=payload,
        )

    def _stage_metrics(self) -> StageReport:
        cfg = self.config.metrics
        metrics: dict[str, Any] = {}

        selected_csv = self.run_dir / "preprocess" / "selected_frames.csv"
        if selected_csv.is_file():
            records = load_frames_csv(selected_csv)
            sharpness = [record.quality.sharpness for record in records if record.quality]
            metrics["frames"] = {
                "count": len(records),
                "duration_s": round(records[-1].timestamp_s - records[0].timestamp_s, 3)
                if len(records) > 1
                else 0.0,
                "with_gps": sum(1 for record in records if record.lat is not None),
                "sharpness_mean": round(float(np.mean(sharpness)), 3) if sharpness else None,
            }

        clouds: dict[str, Any] = {}
        reference = load_ply(cfg.reference_cloud).points if cfg.reference_cloud else None
        dense_payload = self._stage_result("dense") or {}
        sfm_payload = self._stage_result("sfm") or {}
        cloud_candidates: list[tuple[str, Path | None]] = [
            ("georeferenced_sparse", self.run_dir / "georef" / "georeferenced_sparse.ply"),
            ("georeferenced_dense", self.run_dir / "georef" / "georeferenced_dense.ply"),
            ("dense", Path(dense_payload["fused_ply"]) if dense_payload.get("fused_ply") else None),
            ("sparse", Path(sfm_payload["sparse_ply"]) if sfm_payload.get("sparse_ply") else None),
        ]
        for label, path in cloud_candidates:
            if path is None or not Path(path).is_file() or label in clouds:
                continue
            try:
                points = load_ply(path).points
                clouds[label] = summarize_cloud(
                    points, voxel_size=cfg.voxel_size, reference=reference
                )
                clouds[label]["path"] = str(path)
            except Drone3DError as exc:
                log.warning("could not summarize cloud %s: %s", path, exc)
        if clouds:
            metrics["clouds"] = clouds

        georef = self._stage_result("georef")
        if georef:
            metrics["georeferencing"] = georef.get("gps_accuracy", {})
            if cfg.expected_extent_m and "georeferenced_sparse" in clouds:
                measured = clouds["georeferenced_sparse"]["bounds"]["max_extent_m"]
                metrics["scale_check"] = geometric_scale_error(measured, cfg.expected_extent_m)
            elif cfg.expected_extent_m:
                metrics["scale_check"] = {
                    "expected_m": cfg.expected_extent_m,
                    "measured_m": None,
                    "note": "no georeferenced cloud available",
                }

        metrics_path = _write_json(self._stage_dir("metrics") / "metrics.json", metrics)
        return StageReport(
            "metrics",
            "ok",
            f"{len(clouds)} cloud(s) summarised",
            artifacts=[Artifact("metrics", metrics_path)],
            metrics=metrics,
        )

    def _stage_report(self) -> StageReport:
        cfg = self.config.metrics
        stage_dir = self._stage_dir("report")

        contact_sheet: Path | None = None
        if cfg.make_contact_sheet:
            selected_csv = self.run_dir / "preprocess" / "selected_frames.csv"
            frame_paths: list[Path] = []
            if selected_csv.is_file():
                frame_paths = [record.path for record in load_frames_csv(selected_csv)]
            if frame_paths:
                contact_sheet = make_contact_sheet(
                    frame_paths,
                    stage_dir / "contact_sheet.jpg",
                    max_images=cfg.report_thumbnails,
                )

        report_path = generate_report(
            run_dir=self.run_dir,
            result=self._result,
            context={
                "video": _read_json(self.run_dir / "ingest" / "video_info.json") or {},
                "telemetry": _read_json(self.run_dir / "ingest" / "telemetry_summary.json") or {},
                "metrics": self._collect_metrics(),
                "contact_sheet": contact_sheet,
                "config": self.config.to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "version": __version__,
            },
            out_path=self.run_dir / "report.html",
        )
        return StageReport(
            "report",
            "ok",
            "HTML report generated",
            artifacts=[
                Artifact("report", report_path),
                Artifact("manifest", self.run_dir / "manifest.json"),
            ],
        )

    # ------------------------------------------------------------------ helpers

    def _collect_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for stage in self._result.stages:
            if stage.metrics:
                metrics[stage.name] = stage.metrics
        stored = _read_json(self.run_dir / "metrics" / "metrics.json")
        if stored:
            metrics["summary"] = stored
        return metrics

    def _write_manifest(self) -> Path:
        return _write_json(
            self.run_dir / "manifest.json",
            {
                "generator": f"drone3d {__version__}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(self.run_dir),
                "config": self.config.to_dict(),
                "result": self._result.to_dict(),
            },
        )

    # ------------------------------------------------------------------ extras

    def point_cloud_bounds(self, path: str | Path) -> dict[str, Any]:
        """Convenience accessor used by notebooks and the API layer."""
        return cloud_bounds(load_ply(path).points)
