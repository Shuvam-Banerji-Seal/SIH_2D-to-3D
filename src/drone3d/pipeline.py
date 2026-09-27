"""End-to-end orchestration: single-pass drone video -> splats, mesh, georeferenced outputs.

Stages communicate only through files in the run directory, so any stage can
be re-run on its own (``drone3d run --run-dir R --stages splat,mesh``)::

    ingest     probe video, load telemetry
    keyframes  GPU pass segmentation + overlap-band keyframes + 3D verdict
    sfm        spirula-studio SfM, one COLMAP model per reconstructable pass
    depth      Marigold v2 depth, aligned to and scored against SfM
    splat      3D Gaussian Splatting per model, depth-supervised, held-out eval
    mesh       mesh extraction from the splats
    georef     GPS alignment (ground-levelled on straight tracks)
    render     fly-through video of each model
    metrics    everything measurable in one metrics.json
    report     standalone report.html + manifest.json
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from drone3d.config import ALL_STAGES, PipelineConfig
from drone3d.exceptions import BackendUnavailable, Drone3DError
from drone3d.logging_utils import get_logger
from drone3d.types import Artifact, PipelineResult, StageReport
from drone3d.version import __version__

__all__ = ["Pipeline", "default_run_dir"]

log = get_logger(__name__)


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
    """Runs pipeline stages against a single run directory."""

    def __init__(self, config: PipelineConfig, run_dir: str | Path | None = None) -> None:
        self.config = config
        self.run_dir = Path(run_dir) if run_dir is not None else default_run_dir(config)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = self.run_dir / "dataset"
        self._result = PipelineResult(run_dir=self.run_dir)
        self.cancel: threading.Event | None = None  # set by the engine to stop before the next stage

    # ------------------------------------------------------------------ public

    def run(self, stages: list[str] | None = None) -> PipelineResult:
        """Execute ``stages`` (default: configured stages) in order."""
        selected = list(stages or self.config.stages)
        unknown = [stage for stage in selected if stage not in ALL_STAGES]
        if unknown:
            raise Drone3DError(f"unknown stage(s): {', '.join(unknown)}")
        self._result = PipelineResult(run_dir=self.run_dir)
        for name in selected:
            if self.cancel is not None and self.cancel.is_set():
                log.info("stage %s: skipped (0.0s) cancelled", name)
                self._result.stages.append(StageReport(name, "skipped", "cancelled"))
                continue
            self._result.stages.append(self._run_stage(name))
        self._result.metrics = self._collect_metrics()
        log.info("manifest written: %s", self._write_manifest())
        return self._result

    # ----------------------------------------------------------- stage plumbing

    def _run_stage(self, name: str) -> StageReport:
        handler = getattr(self, f"_stage_{name}")
        log.info("=== stage: %s ===", name)
        started = time.perf_counter()
        monitor = None
        if self.config.metrics.gpu_telemetry and name not in ("ingest", "metrics", "report"):
            from drone3d.gpu.monitor import GpuMonitor

            monitor = GpuMonitor(interval=0.5).__enter__()
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
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)
        report.duration_s = time.perf_counter() - started
        if monitor is not None:
            gpu = monitor.summary()
            if gpu.samples:
                report.metrics["gpu"] = gpu.to_dict()
                _write_json(
                    self._stage_dir(name) / "gpu_timeline.json", gpu.to_dict(with_timeline=True)
                )
                stored = self._stage_result(name)
                if stored is not None:
                    stored["gpu"] = gpu.to_dict()
                    stored["duration_s"] = round(report.duration_s, 2)
                    _write_json(self._stage_dir(name) / "result.json", stored)
        log.info("stage %s: %s (%.1fs) %s", name, report.status, report.duration_s, report.message)
        return report

    def _stage_dir(self, stage: str) -> Path:
        path = self.run_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _stage_result(self, stage: str) -> dict[str, Any] | None:
        return _read_json(self.run_dir / stage / "result.json")

    def _video_info(self):  # type: ignore[no-untyped-def]
        from drone3d.io.nvdec import probe_stream

        if not self.config.ingest.video:
            raise BackendUnavailable("ingest.video is not configured")
        return probe_stream(self.config.ingest.video)

    def _sfm_models(self, min_images: int = 0) -> list[dict[str, Any]]:
        sfm = self._stage_result("sfm") or {}
        return [m for m in sfm.get("models", []) if (m.get("images") or 0) >= min_images]

    # ------------------------------------------------------------------- stages

    def _stage_ingest(self) -> StageReport:
        from drone3d.io.telemetry import load_telemetry, telemetry_summary

        cfg = self.config.ingest
        info = self._video_info()
        samples = load_telemetry(cfg.telemetry) if cfg.telemetry else []
        for sample in samples:  # the video starts this far into the log (a live segment)
            sample.t -= cfg.telemetry_offset_s
        stage_dir = self._stage_dir("ingest")
        if samples:
            _write_json(stage_dir / "telemetry.json", [s.to_dict() for s in samples])
        payload = {"video": info.to_dict(), "telemetry": telemetry_summary(samples)}
        _write_json(stage_dir / "result.json", payload)
        return StageReport(
            "ingest",
            "ok",
            f"{info.width}x{info.height} {info.codec} @ {info.fps:.2f} fps, "
            f"{info.num_frames} frames, {len(samples)} telemetry samples",
            artifacts=[Artifact("ingest_result", stage_dir / "result.json")],
            metrics=payload,
        )

    def _stage_keyframes(self) -> StageReport:
        import torch

        from drone3d.engine import models
        from drone3d.io.nvdec import (
            analysis_size,
            detect_letterbox,
            extract_frames,
            letterbox_from_frames,
            stream_analysis_chunks,
            write_sidecar_jpegs,
        )
        from drone3d.keyframes.flow import ConsecutiveFlow
        from drone3d.keyframes.select import SelectorConfig, select_keyframes

        cfg = self.config.keyframes
        info = self._video_info()
        stage_dir = self._stage_dir("keyframes")
        stride = max(1, round(info.fps / cfg.analysis_fps))
        size = analysis_size(info.width, info.height, cfg.analysis_long_side)
        # Letterbox bars are static and black: left in, they count as perfectly
        # tracked content (inflating overlap), get depth predicted for them and
        # inflate held-out PSNR with trivially correct pixels.
        t0 = time.perf_counter()
        flow_model = models.raft(cfg.flow_model, batch=cfg.flow_batch)
        probe, crop = None, None
        # One set of samples serves both the letterbox test and the motion probe (each ffmpeg
        # sample costs ~0.2 s; two sets of twelve were 11 s of a 76 s Jal Mahal run).
        pairs = _probe_pairs(info, size, stride) if cfg.adaptive_rate else None
        if cfg.crop_letterbox:
            crop = (letterbox_from_frames(info, [p[0] for p in pairs], size) if pairs and len(pairs) >= 4
                    else detect_letterbox(info))  # fmt: skip
        if cfg.adaptive_rate:
            stride, probe = _adapt_stride(info, size, stride, flow_model, crop, cfg, pairs=pairs)
        # One decode for analysis and keyframes when the keyframes are small enough to write
        # every analysis frame as a candidate (the default profile keeps full-size keyframes).
        side = (
            analysis_size(info.width, info.height, cfg.output_long_side, multiple=2)
            if cfg.output_long_side and cfg.output_long_side <= 2048
            else None
        )
        candidates = self.dataset / "candidates"
        if candidates.exists():
            shutil.rmtree(candidates)
        chunks = stream_analysis_chunks(info, size, stride=stride, hwaccel=cfg.hwaccel, crop=crop, sidecar=side)
        if side is not None:
            chunks = write_sidecar_jpegs(chunks, candidates, quality=cfg.jpeg_quality)
        capacity = -(-info.num_frames // stride)  # ceil: analysis frames expected
        flow, indices, thumbs = ConsecutiveFlow.compute(chunks, flow_model, capacity=capacity)
        t1 = time.perf_counter()
        selector = SelectorConfig(
            overlap_target=cfg.overlap_target,
            overlap_band=cfg.overlap_band,
            max_gap_s=cfg.max_gap_s,
            min_pass_s=cfg.min_pass_s,
            cut_consistency=cfg.cut_consistency,
            hfov_deg=cfg.hfov_deg,
            parallax_snr=cfg.parallax_snr,
            relative_overlap=cfg.relative_overlap,
        )
        selection = select_keyframes(flow, indices, info.fps, selector, flow_model=flow_model)
        t2 = time.perf_counter()
        kept_passes = [
            p.pass_id
            for p in selection.passes
            if (not cfg.passes or p.pass_id in cfg.passes)
            and not (cfg.skip_degenerate and p.verdict in ("degenerate", "too-short"))
        ]
        keyframes = [k for k in selection.keyframes if k.pass_id in kept_passes]
        analysis = {
            "frames": int(flow.num_frames),
            "stride": stride,
            "rate_probe": probe,
            "size": list(size),
            "flow_model": cfg.flow_model,
            "flows": 2 * (flow.num_frames - 1),
            "flows_per_second": round(2 * (flow.num_frames - 1) / max(t1 - t0, 1e-9), 1),
        }
        # Release the analysis state (flows, frames, CUDA-graph pools) before
        # decoding 4K keyframes, which needs the memory.
        del flow, flow_model, chunks
        torch.cuda.empty_cache()
        if not keyframes:
            raise Drone3DError(
                "no pass with recoverable 3D structure (see keyframes/selection.json)"
            )

        images = self.dataset / "images"
        if images.exists():
            shutil.rmtree(images)  # a re-run must not mix in stale keyframes
        names = [f"pass_{k.pass_id:02d}/f_{k.frame_index:06d}.jpg" for k in keyframes]
        if side is not None:  # written during the analysis decode: keep the chosen ones
            for k, n in zip(keyframes, names, strict=True):
                (images / n).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(candidates / f"f_{k.frame_index:06d}.jpg", images / n)
            shutil.rmtree(candidates)
        else:
            extract_frames(
                info,
                [k.frame_index for k in keyframes],
                [images / n for n in names],
                long_side=cfg.output_long_side,
                quality=cfg.jpeg_quality,
                hwaccel=cfg.hwaccel,
                crop=crop,
            )
        t3 = time.perf_counter()

        rows = self._keyframe_rows(keyframes, names)
        _write_json(stage_dir / "keyframes.json", rows)
        _write_json(stage_dir / "selection.json", selection.to_dict())
        torch.save(thumbs.cpu(), stage_dir / "thumbs.pt")
        figures = []
        try:
            from drone3d.report.figures import keyframe_timeline

            figures.append(
                str(keyframe_timeline(selection.to_dict(), stage_dir / "keyframe_timeline.png"))
            )
        except Exception as exc:  # figures are diagnostics, never fatal
            log.warning("keyframe figure failed: %s", exc)
        passes = [p.__dict__ for p in selection.passes]
        payload = {
            "crop": list(crop) if crop else None,
            "analysis": analysis,
            "timing_s": {
                "decode_and_flow": round(t1 - t0, 2),
                "selection": round(t2 - t1, 2),
                "extraction": round(t3 - t2, 2),
            },
            "passes": passes,
            "kept_passes": kept_passes,
            "num_keyframes": len(keyframes),
            "num_keyframes_all_passes": len(selection.keyframes),
            "with_gps": sum(1 for r in rows if r.get("lat") is not None),
            "figures": figures,
        }
        _write_json(stage_dir / "result.json", payload)
        verdicts = ", ".join(f"pass {p.pass_id}: {p.verdict}" for p in selection.passes)
        return StageReport(
            "keyframes",
            "ok",
            f"{len(keyframes)} keyframes from {flow_frames_str(payload)} ({verdicts})",
            artifacts=[
                Artifact("keyframe_images", images, "directory"),
                Artifact("selection", stage_dir / "selection.json"),
            ],
            metrics=payload,
        )

    def _keyframe_rows(self, keyframes: list, names: list[str]) -> list[dict[str, Any]]:
        """Keyframe table with GPS interpolated from telemetry by timestamp."""
        from drone3d.io.telemetry import interpolate_sample
        from drone3d.types import TelemetrySample

        raw = _read_json(self.run_dir / "ingest" / "telemetry.json") or []
        samples = [
            TelemetrySample(
                **{k: v for k, v in s.items() if k in TelemetrySample.__dataclass_fields__}
            )
            for s in raw
        ]
        rows = []
        for k, name in zip(keyframes, names, strict=True):
            row = {"name": name, **k.__dict__}
            sample = interpolate_sample(samples, k.timestamp_s) if samples else None
            if sample is not None and sample.has_position():
                row.update(
                    lat=sample.lat,
                    lon=sample.lon,
                    alt_m=sample.alt_m if sample.alt_m is not None else sample.rel_alt_m,
                )
            rows.append(row)
        return rows

    def _stage_sfm(self) -> StageReport:
        cfg = self.config.sfm
        if cfg.backend == "none":
            return StageReport("sfm", "skipped", "sfm.backend is none")
        images = self.dataset / "images"
        if not images.is_dir():
            return StageReport("sfm", "skipped", "no keyframes (run the keyframes stage first)")
        if cfg.backend == "flow":
            return self._stage_sfm_flow()
        from drone3d.splat.spirula import run_sfm

        sequences = sorted(p.name for p in images.iterdir() if p.is_dir())
        result = run_sfm(
            self.dataset,
            sequences=sequences,
            quality=cfg.quality,
            camera_mode=cfg.camera_mode,
            camera_model=cfg.camera_model,
            features=cfg.features,
            extra=cfg.extra_args,
            log_path=self._stage_dir("sfm") / "spirula_sfm.log",
        )
        for model in result.models:
            model["passes"] = _model_passes(Path(model["path"]))
        payload = result.to_dict()
        _write_json(self._stage_dir("sfm") / "result.json", payload)
        best = result.models[0] if result.models else {}
        return StageReport(
            "sfm",
            "ok" if result.models else "failed",
            f"{result.registered_images}/{result.input_images} keyframes registered in "
            f"{len(result.models)} model(s); largest {best.get('images')} images, "
            f"{result.mean_reprojection_px} px mean reprojection",
            artifacts=[Artifact("sfm_result", self._stage_dir("sfm") / "result.json")],
            metrics=payload,
        )

    def _stage_sfm_flow(self) -> StageReport:
        cfg = self.config.sfm
        from drone3d.fastsfm.stage import run_flow_sfm

        payload = run_flow_sfm(
            self.dataset,
            self._stage_dir("sfm") / "work",
            long_side=cfg.flow_long_side,
            span=cfg.flow_span,
            stride=cfg.flow_stride,
            max_gap=cfg.flow_max_gap,
            mapper=cfg.mapper,
            hfov_deg=self.config.keyframes.hfov_deg,
        )
        _write_json(self._stage_dir("sfm") / "result.json", payload)
        best = payload["models"][0] if payload["models"] else {}
        return StageReport(
            "sfm",
            "ok" if payload["models"] else "failed",
            f"flow tracks: {payload['registered_images']}/{payload['input_images']} keyframes registered in "
            f"{len(payload['models'])} model(s); largest {best.get('images')} images, "
            f"{payload['mean_reprojection_px']} px mean reprojection",
            artifacts=[Artifact("sfm_result", self._stage_dir("sfm") / "result.json")],
            metrics=payload,
        )

    def _stage_dense(self) -> StageReport:
        cfg = self.config.dense
        if cfg.backend == "none":
            return StageReport("dense", "skipped", "dense.backend is none")
        models = self._sfm_models(cfg.min_model_images)
        if not models:
            return StageReport("dense", "skipped", "no SfM model (run sfm first)")
        from drone3d.fastsfm.dense_stage import run_dense

        payload = run_dense(
            [Path(m["path"]) for m in models],
            self.dataset / "images",
            self._stage_dir("dense"),
            long_side=cfg.long_side,
            gaps=tuple(cfg.gaps),
            keyframe_stride=cfg.keyframe_stride,
            min_angle_deg=cfg.min_angle_deg,
            rel_tol=cfg.rel_tol,
            mono_model=cfg.mono_model,
            voxel_px=cfg.voxel_px,
            trunc_voxels=cfg.trunc_voxels,
            tsdf_memory_gb=cfg.tsdf_memory_gb,
            isolate_fusion=cfg.isolate_fusion,
            refine=cfg.refine,
            min_views=cfg.min_views,
        )
        _write_json(self._stage_dir("dense") / "result.json", payload)
        ok = [m for m in payload["models"] if m.get("status") == "ok"]
        return StageReport(
            "dense",
            "ok" if ok else "failed",
            "; ".join(
                f"{Path(m['model']).name}: {m['mesh_triangles']:,} triangles, {m['num_points']:,} points, "
                f"depth on {100 * m['coverage']:.0f} % of pixels"
                for m in ok
            )
            or "no model produced depth",
            metrics=payload,
        )

    def _stage_export(self) -> StageReport:
        cfg = self.config.export
        if not cfg.enabled:
            return StageReport("export", "skipped", "export.enabled is false")
        dense = self._stage_result("dense")
        if not dense or not any(m.get("status") == "ok" for m in dense.get("models", [])):
            return StageReport("export", "skipped", "no dense model (run dense first)")
        from drone3d.export.stage import run_export

        payload = run_export(
            dense,
            self._stage_result("georef"),
            self._stage_dir("export"),
            title=self.config.run_name,
            mesh_formats=list(cfg.mesh_formats),
            las=cfg.las,
            geotiff=cfg.geotiff,
            raster_cell=cfg.raster_cell,
            viewer=cfg.viewer,
            images=self.dataset / "images",
            texture=cfg.texture,
            texture_views=cfg.texture_views,
            texture_size=cfg.texture_size,
            max_triangles=cfg.max_triangles,
            splats=self._stage_result("splat") if cfg.splats else None,
            max_splats=cfg.max_splats,
            texture_gain=cfg.texture_gain,
        )
        _write_json(self._stage_dir("export") / "result.json", payload)
        n_files = sum(len(m["files"]) for m in payload["models"])
        return StageReport(
            "export",
            "ok",
            f"{n_files} files for {len(payload['models'])} model(s)"
            + (f"; viewer: drone3d view {self.run_dir}" if payload.get("viewer") else ""),
            metrics=payload,
        )

    def _stage_depth(self) -> StageReport:
        cfg = self.config.depth
        if cfg.backend == "none":
            return StageReport("depth", "skipped", "depth.backend is none")
        models = self._sfm_models(self.config.splat.min_model_images)
        if not models:
            return StageReport("depth", "skipped", "no SfM model (run sfm first)")
        from drone3d.depth.stage import run_depth

        for stale in ("depths", "depth_raw"):
            if (self.dataset / stale).exists():
                shutil.rmtree(self.dataset / stale)
        result = run_depth(
            self.dataset,
            [Path(m["path"]) for m in models],
            long_side=cfg.long_side,
            batch=cfg.batch,
            quantization=cfg.quantization,
            far_factor=cfg.far_factor,
            out_long_side=cfg.out_long_side,
            calibration=cfg.calibration,
        )
        payload = result.to_dict()
        _write_json(self._stage_dir("depth") / "result.json", payload)
        agree = [
            v.get("abs_rel_median")
            for v in payload["per_model"].values()
            if v.get("abs_rel_median") is not None
        ]
        return StageReport(
            "depth",
            "ok",
            f"Marigold v2 on {result.images} keyframes ({payload['images_per_second']} img/s); "
            f"median AbsRel vs SfM {min(agree) if agree else 'n/a'}-{max(agree) if agree else 'n/a'}",
            metrics=payload,
        )

    def _stage_splat(self) -> StageReport:
        cfg = self.config.splat
        if cfg.backend == "none":
            return StageReport("splat", "skipped", "splat.backend is none")
        from drone3d.splat.spirula import run_train

        models = self._sfm_models(cfg.min_model_images)
        if not models:
            return StageReport("splat", "skipped", "no SfM model with enough images")
        if cfg.models == "largest":
            models = models[:1]
        have_depth = (self.dataset / "depths").is_dir()
        dense = {m["model"]: m for m in (self._stage_result("dense") or {}).get("models", []) if m.get("status") == "ok"}
        def train(model: dict) -> dict:
            recon = Path(model["path"]).relative_to(self.dataset)
            if cfg.init == "dense" and model["path"] in dense:  # start from the TSDF cloud, not ~1000 SfM points
                from drone3d.splat.spirula import dense_init_model

                init_dir = self.dataset / "sparse_init" / Path(model["path"]).name
                dense_init_model(Path(model["path"]), Path(dense[model["path"]]["points"]), init_dir,
                                 max_points=cfg.init_points)  # fmt: skip
                recon = init_dir.relative_to(self.dataset)
            name = f"model_{Path(model['path']).name}"
            flags = {
                "train_resolution_divisor": cfg.resolution_divisor,
                "cache_images": cfg.cache_images,
                **cfg.flags,
            }
            if not have_depth:
                flags["load_depths"] = 0
            result = run_train(
                self.dataset,
                self.run_dir / "splats" / name,
                preset=cfg.preset,
                recon_dir=str(recon),
                iterations=cfg.iterations,
                quality=cfg.quality,
                depth_weight=cfg.depth_weight if have_depth else 0.0,
                eval_interval=cfg.eval_interval,
                flags=flags,
            )
            return {"model": model["path"], "passes": model.get("passes"), **result.to_dict()}

        if cfg.parallel > 1 and len(models) > 1:  # one trainer leaves the GPU idle between its small kernels
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(min(cfg.parallel, len(models)), thread_name_prefix="splat") as pool:
                trained = list(pool.map(train, models))  # model order kept
        else:
            trained = [train(m) for m in models]
        payload = {"models": trained, "depth_supervised": have_depth and cfg.depth_weight > 0}
        _write_json(self._stage_dir("splat") / "result.json", payload)
        summary = "; ".join(
            f"{Path(t['run_dir']).name}: PSNR {_mean_metric(t['eval_metrics'], 'psnr'):.2f} dB, "
            f"{t['num_splats']} splats"
            for t in trained
        )
        return StageReport("splat", "ok", summary, metrics=payload)

    def _stage_mesh(self) -> StageReport:
        cfg = self.config.mesh
        if cfg.backend == "none":
            return StageReport("mesh", "skipped", "mesh.backend is none")
        splat = self._stage_result("splat")
        if not splat or not splat.get("models"):
            return StageReport("mesh", "skipped", "no trained splats (run splat first)")
        from drone3d.splat.spirula import run_mesh

        meshes = []
        for model in splat["models"]:
            result = run_mesh(
                Path(model["run_dir"]),
                formats=cfg.formats,
                colors=cfg.colors,
                flags={"num_threads": cfg.num_threads},
            )
            meshes.append(
                {"run_dir": model["run_dir"], **result.to_dict(), **_mesh_stats(result.files)}
            )
        payload = {"meshes": meshes}
        _write_json(self._stage_dir("mesh") / "result.json", payload)
        return StageReport(
            "mesh",
            "ok",
            "; ".join(
                f"{Path(m['run_dir']).name}: {m.get('vertices')} vertices / {m.get('faces')} faces"
                for m in meshes
            ),
            metrics=payload,
        )

    def _stage_georef(self) -> StageReport:
        cfg = self.config.geo
        if not cfg.enabled:
            return StageReport("georef", "skipped", "geo.enabled is false")
        rows = _read_json(self.run_dir / "keyframes" / "keyframes.json") or []
        gps = (
            {r["name"]: r for r in rows if r.get("lat") is not None}
            if isinstance(rows, list)
            else {}
        )
        if not gps:
            return StageReport(
                "georef", "skipped", "keyframes carry no GPS (ingest.telemetry missing?)"
            )
        import pycolmap

        from drone3d.geo.georef import solve_georef
        from drone3d.geo.projection import LocalTangentPlane
        from drone3d.utils.ply import write_ply

        lat0 = (
            cfg.origin_lat
            if cfg.origin_lat is not None
            else float(np.median([r["lat"] for r in gps.values()]))
        )
        lon0 = (
            cfg.origin_lon
            if cfg.origin_lon is not None
            else float(np.median([r["lon"] for r in gps.values()]))
        )
        alts = [r["alt_m"] for r in gps.values() if r.get("alt_m") is not None]
        alt0 = float(np.median(alts)) if alts else cfg.origin_alt
        plane = LocalTangentPlane(lat0, lon0, alt0)
        stage_dir = self._stage_dir("georef")
        results = []
        for model in self._sfm_models(cfg.min_correspondences):
            rec = pycolmap.Reconstruction(model["path"])
            pairs = [
                (im, gps[im.name]) for im in rec.images.values() if im.has_pose and im.name in gps
            ]
            if len(pairs) < cfg.min_correspondences:
                continue
            centers = np.array([im.projection_center() for im, _ in pairs])
            enu = np.array(
                [plane.to_local(r["lat"], r["lon"], r.get("alt_m") or alt0) for _, r in pairs]
            )
            points = np.array([p.xyz for p in rec.points3D.values()])
            colors = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
            transform, info = solve_georef(centers, enu, points, mode=cfg.align_mode)
            name = Path(model["path"]).name
            cloud = stage_dir / f"model_{name}_sparse_enu.ply"
            write_ply(
                cloud,
                transform.apply(points),
                colors,
                comment=f"ENU metres; origin WGS84 {lat0:.8f} {lon0:.8f} {alt0:.3f}",
            )
            results.append(
                {
                    "model": model["path"],
                    "transform": transform.to_dict(),
                    **info,
                    "cloud": str(cloud),
                }
            )
        if not results:
            return StageReport(
                "georef",
                "skipped",
                f"fewer than {cfg.min_correspondences} GPS-tagged keyframes per model",
            )
        payload = {"origin": plane.to_dict(), "models": results}
        if cfg.write_geojson:
            _write_json(
                stage_dir / "camera_track.geojson",
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                         "properties": {"name": n, "alt_m": r.get("alt_m")}}
                        for n, r in sorted(gps.items())
                    ],
                },
            )  # fmt: skip
        _write_json(stage_dir / "result.json", payload)
        best = results[0]
        return StageReport(
            "georef",
            "ok",
            f"{best['mode']} fit; held-out horizontal RMSE "
            f"{best.get('held_out', {}).get('loo_rmse_horizontal_m', float('nan')):.2f} m",
            metrics=payload,
        )

    def _stage_render(self) -> StageReport:
        cfg = self.config.render
        if not cfg.enabled:
            return StageReport("render", "skipped", "render.enabled is false")
        splat = self._stage_result("splat")
        if not splat or not splat.get("models"):
            return StageReport("render", "skipped", "no trained splats (run splat first)")
        from drone3d.splat.render import render_flythrough

        videos = []
        for model in splat["models"]:
            if not model.get("splat_ply"):
                continue
            out = self._stage_dir("render") / f"{Path(model['run_dir']).name}_flythrough.mp4"
            videos.append(
                render_flythrough(
                    Path(model["splat_ply"]),
                    Path(model["model"]),
                    out,
                    seconds=cfg.seconds,
                    fps=cfg.fps,
                    long_side=cfg.long_side,
                    scene_transform=Path(model["run_dir"]) / "scene_transform.json",
                )
            )
        payload = {"videos": videos}
        _write_json(self._stage_dir("render") / "result.json", payload)
        return StageReport("render", "ok", f"{len(videos)} fly-through video(s)", metrics=payload)

    def _stage_metrics(self) -> StageReport:
        metrics = _summary_metrics(self.run_dir)
        clouds = self._cloud_metrics()
        if clouds:
            metrics["clouds"] = clouds
        path = _write_json(self._stage_dir("metrics") / "metrics.json", metrics)
        return StageReport(
            "metrics",
            "ok",
            f"{len(metrics)} metric groups",
            artifacts=[Artifact("metrics", path)],
            metrics=metrics,
        )

    def _cloud_metrics(self) -> list[dict[str, Any]]:
        """Per exported model: bounds and voxel coverage, and accuracy/completeness vs ``metrics.reference_cloud``.

        A LAS/LAZ reference is compared in the model's UTM zone (so only
        georeferenced models are compared); any other format is taken to be in
        the export frame of ``points.ply``.
        """
        import open3d as o3d

        from drone3d.export.stage import _enu_to_utm
        from drone3d.metrics.quality import load_cloud, summarize_cloud

        export = self._stage_result("export")
        if not export:
            return []
        cfg = self.config.metrics
        origin = (self._stage_result("georef") or {}).get("origin")
        ref_path = Path(cfg.reference_cloud) if cfg.reference_cloud else None
        projected = ref_path is not None and ref_path.suffix.lower() in (".las", ".laz")
        refs: dict[int | None, Any] = {}
        rows = []
        for m in export.get("models", []):
            ply = next((f for f in m.get("files", []) if f.endswith("points.ply")), None)
            if ply is None:
                continue
            p = np.asarray(o3d.io.read_point_cloud(str(self.run_dir / "export" / ply)).points, dtype=np.float64)
            if not len(p):
                continue
            row: dict[str, Any] = {"model": m["model"], "units": m.get("units"), "epsg": m.get("epsg")}
            reference, epsg = None, m.get("epsg")
            if ref_path is not None:
                if projected and not (epsg and origin):
                    row["reference"] = "not compared: a LAS reference needs a georeferenced model"
                else:
                    key = epsg if projected else None
                    if key not in refs:
                        refs[key] = load_cloud(ref_path, target_epsg=key)
                    reference = refs[key]
                    if projected:
                        p = _enu_to_utm(p, origin, epsg)
                    row["reference"] = str(ref_path)
            row.update(summarize_cloud(p, voxel_size=cfg.voxel_size, reference=reference,
                                       distance_threshold_m=cfg.reference_threshold_m))  # fmt: skip
            rows.append(row)
        return rows

    def _stage_report(self) -> StageReport:
        from drone3d.report.html import generate_report, make_contact_sheet

        stage_dir = self._stage_dir("report")
        frames = (
            sorted((self.dataset / "images").rglob("*.jpg"))
            if (self.dataset / "images").is_dir()
            else []
        )
        contact = (
            make_contact_sheet(
                frames,
                stage_dir / "contact_sheet.jpg",
                max_images=self.config.metrics.report_thumbnails,
            )
            if frames
            else None
        )
        ingest = self._stage_result("ingest") or {}
        report_path = generate_report(
            run_dir=self.run_dir,
            result=self._result,
            context={
                "video": ingest.get("video", {}),
                "telemetry": ingest.get("telemetry", {}),
                "metrics": self._collect_metrics(),
                "contact_sheet": contact,
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
        """Every stage's persisted result, overlaid with this invocation's reports."""
        metrics: dict[str, Any] = {}
        for name in ALL_STAGES:
            if name in ("metrics", "report"):
                continue
            stored = self._stage_result(name)
            if stored:
                metrics[name] = stored
        for stage in self._result.stages:
            if stage.metrics and stage.name not in ("metrics", "report"):
                metrics[stage.name] = stage.metrics
        summary = _read_json(self.run_dir / "metrics" / "metrics.json")
        if summary:
            metrics["summary"] = summary
        return metrics

    def _write_manifest(self) -> Path:
        """Describe the whole run directory, not only this invocation's stages."""
        stages = [stage.to_dict() for stage in self._result.stages]
        seen = {stage["name"] for stage in stages}
        for name in ALL_STAGES:
            stored = self._stage_result(name)
            if name in seen or not stored:
                continue
            stages.append(
                {
                    "name": name,
                    "status": "ok",
                    "message": "",
                    "duration_s": stored.get("duration_s", 0.0),
                    "metrics": stored,
                }
            )
        return _write_json(
            self.run_dir / "manifest.json",
            {
                "generator": f"drone3d {__version__}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(self.run_dir),
                "config": self.config.to_dict(),
                "result": {**self._result.to_dict(), "stages": stages},
            },
        )


def flow_frames_str(payload: dict[str, Any]) -> str:
    a = payload["analysis"]
    return f"{a['frames']} analysis frames at {a['flows_per_second']} flows/s"


def _probe_pairs(info, size, stride):  # type: ignore[no-untyped-def]
    """Twelve frame pairs one analysis step apart, spread over the video (none for clips under 10 s)."""
    from drone3d.io.nvdec import sample_pairs

    if info.duration_s < 10:
        return None
    return sample_pairs(info, np.linspace(0.05, 0.9, 12) * info.duration_s, size, stride)


def _adapt_stride(info, size, stride, flow_model, crop, cfg, pairs=None):  # type: ignore[no-untyped-def]
    """Grow the analysis stride while the camera moves slowly.

    Twelve frame pairs one nominal step apart, spread over the video, give the
    median flow per step; the stride is multiplied until a step moves about
    ``target_motion`` of the width (never below ``min_analysis_fps``). A survey
    pass drifting slowly across the ground then costs a fraction of the flows a
    fast cinematic orbit needs, which is where a 10-minute video's time goes.
    """
    import torch

    from drone3d.io.nvdec import _color_args, nv12_to_rgb, scaled_crop

    width, height = size
    if info.duration_s < 10:
        return stride, None
    if pairs is None:
        pairs = _probe_pairs(info, size, stride)
    if len(pairs) < 4:
        return stride, {"status": "too-few-samples"}
    nv = torch.from_numpy(np.stack([f for p in pairs for f in p])).view(-1, height * 3 // 2, width).cuda()
    rgb = nv12_to_rgb(nv, height, width, **_color_args(info))
    if crop is not None:
        x0, y0, x1, y1 = scaled_crop(crop, (info.width, info.height), size, multiple=8)
        rgb = rgb[:, y0:y1, x0:x1].contiguous()
    flow = flow_model(rgb[0::2].contiguous(), rgb[1::2].contiguous())
    per_pair = flow.norm(dim=1).flatten(1).median(dim=1).values.cpu().numpy()
    motion = float(np.median(per_pair))
    target = cfg.target_motion * rgb.shape[2]
    max_stride = max(stride, int(info.fps // cfg.min_analysis_fps))
    new = int(min(max_stride, stride * max(1, int(target // max(motion, 1e-6)))))
    log.info("analysis rate: %.1f px per step at stride %d (target %.1f px) -> stride %d (%.1f fps)",
             motion, stride, target, new, info.fps / new)  # fmt: skip
    return new, {"pairs": len(pairs), "median_px_per_step": round(motion, 2), "target_px": round(target, 1),
                 "nominal_stride": stride, "stride": new}  # fmt: skip


def _model_passes(model_dir: Path) -> dict[str, int]:
    try:
        import pycolmap
    except ImportError:  # pragma: no cover
        return {}
    rec = pycolmap.Reconstruction(str(model_dir))
    counts: dict[str, int] = {}
    for image in rec.images.values():
        if image.has_pose:
            key = image.name.split("/")[0] if "/" in image.name else "."
            counts[key] = counts.get(key, 0) + 1
    return counts


def _mesh_stats(files: list[Path]) -> dict[str, Any]:
    ply = next((f for f in files if f.suffix == ".ply"), None)
    if ply is None:
        return {}
    from drone3d.utils.ply import read_ply_header

    try:
        header = read_ply_header(ply)
    except Exception:  # stats are best-effort
        return {}
    elements = header.get("elements", {}) if isinstance(header, dict) else {}
    return {
        "vertices": elements.get("vertex", {}).get("count"),
        "faces": elements.get("face", {}).get("count"),
    }


def _mean_metric(metrics: dict, key: str) -> float:
    """A held-out metric as one number: spirula reports ``avg_<key>`` and a per-image list under ``key``."""
    value = metrics.get(f"avg_{key}", metrics.get(key))
    if isinstance(value, list):
        value = float(np.mean(value)) if value else None
    return float("nan") if value is None else float(value)


def _summary_metrics(run_dir: Path) -> dict[str, Any]:
    """Flatten the headline numbers of every stage into one dictionary."""
    out: dict[str, Any] = {}
    kf = _read_json(run_dir / "keyframes" / "result.json")
    if kf:
        out["keyframes"] = {
            "count": kf["num_keyframes"],
            "passes": [{k: p.get(k) for k in ("pass_id", "start_s", "end_s", "num_keyframes", "mean_overlap_consecutive", "views_per_point", "parallax_snr", "verdict")} for p in kf["passes"]],
            "analysis": kf["analysis"],
            "timing_s": kf["timing_s"],
        }  # fmt: skip
    sfm = _read_json(run_dir / "sfm" / "result.json")
    if sfm:
        out["sfm"] = {
            "registered": sfm["registered_images"],
            "input": sfm["input_images"],
            "registration_rate": round(sfm["registered_images"] / max(1, sfm["input_images"]), 4),
            "models": [
                {k: m.get(k) for k in ("images", "points", "mean_track_length", "passes")}
                for m in sfm["models"]
            ],
            "mean_reprojection_px": sfm["mean_reprojection_px"],
            "seconds": sfm["seconds"],
        }
    depth = _read_json(run_dir / "depth" / "result.json")
    if depth:
        out["depth"] = {
            "per_model": depth["per_model"],
            "images_per_second": depth.get("images_per_second"),
        }
    splat = _read_json(run_dir / "splat" / "result.json")
    if splat:
        out["splat"] = [
            {"run": Path(m["run_dir"]).name, "passes": m.get("passes"), "num_splats": m["num_splats"], "seconds": m["seconds"], **{k: round(_mean_metric(m["eval_metrics"], k), 4) for k in ("psnr", "ssim", "lpips", "cc_psnr", "cc_ssim")}}
            for m in splat["models"]
        ]  # fmt: skip
    mesh = _read_json(run_dir / "mesh" / "result.json")
    if mesh:
        out["mesh"] = [
            {k: m.get(k) for k in ("run_dir", "vertices", "faces", "seconds")}
            for m in mesh["meshes"]
        ]
    dense = _read_json(run_dir / "dense" / "result.json")
    if dense:
        out["dense"] = [
            {k: m.get(k) for k in ("model", "keyframes", "coverage_triangulated", "coverage", "view_completeness", "mesh_triangles", "num_points", "voxel")}
            for m in dense["models"]
            if m.get("status") == "ok"
        ]  # fmt: skip
        from drone3d.metrics.quality import scene_view_completeness

        out["view_completeness"] = scene_view_completeness(_read_json(run_dir / "sfm" / "result.json") or {}, dense)
    export = _read_json(run_dir / "export" / "result.json")
    if export:
        out["export"] = {
            "files": sorted(f for m in export["models"] for f in m["files"]),
            "viewer": export.get("viewer"),
        }
    # Problem statement 26158: < 15 minutes of processing for a 10-minute video.
    ingest = _read_json(run_dir / "ingest" / "result.json") or {}
    video_s = (ingest.get("video") or {}).get("duration_s")
    stage_s = {
        st: r["duration_s"]
        for st in ALL_STAGES
        if st not in ("metrics", "report") and (r := _read_json(run_dir / st / "result.json") or {}).get("duration_s")
    }
    if stage_s:
        total = round(sum(stage_s.values()), 1)
        budget = round(1.5 * video_s, 1) if video_s else None
        out["processing"] = {
            "seconds": total,
            "per_stage_s": stage_s,
            "video_seconds": video_s,
            "budget_seconds": budget,
            "within_budget": (total <= budget) if budget else None,
        }
    geo = _read_json(run_dir / "georef" / "result.json")
    if geo and geo.get("models"):
        out["georef"] = [
            {k: m.get(k) for k in ("model", "mode", "in_sample", "held_out", "track")}
            for m in geo["models"]
        ]
        # PS deliverable 5: metric scale and how well GPS pins it down (jackknife
        # over GPS fixes). No GPS means no metric scale, and none is claimed.
        out["scale_check"] = [
            {
                "model": m.get("model"),
                "metres_per_model_unit": (m.get("transform") or {}).get("scale"),
                "relative_std": (m.get("held_out") or {}).get("scale_relative_std"),
            }
            for m in geo["models"]
        ]
    gpu = {}
    for stage in ALL_STAGES:
        stored = _read_json(run_dir / stage / "result.json") or {}
        if "gpu" in stored:
            gpu[stage] = {
                k: stored["gpu"].get(k)
                for k in (
                    "util_mean",
                    "power_mean_w",
                    "memory_peak_gb",
                    "own_memory_peak_gb",
                    "host_rss_peak_gb",
                    "shared_gpu",
                )
            }
            gpu[stage]["duration_s"] = stored.get("duration_s")
    if gpu:
        out["gpu"] = gpu
    return out
