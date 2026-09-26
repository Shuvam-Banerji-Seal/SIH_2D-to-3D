# Roadmap

Status legend: `done` · `scaffolded` (interfaces + config wired, needs backend
or dataset) · `planned`.

## M0 — Repository foundation — done

- uv-first `pyproject.toml`, committed `uv.lock`, Makefile targets
- CLI (`run`, `init-config`, `doctor`, `version`), typed config with `--set`
- CI: ruff lint/format, import + CLI smoke checks, config validation
- Docs: problem statement, architecture, data formats, this roadmap

## M1 — Ingest & preprocessing — scaffolded

- [x] Video probing, deterministic frame sampling, frame manifests
- [x] CSV / DJI SRT / GPX / JSON telemetry parsers + interpolation
- [x] Quality scoring (sharpness, exposure, contrast) and keyframe selection
- [x] Focus-aware deblur, stabilization, motion/YOLO dynamic masks
- [x] Unit tests: frame sampling + `max_frames` budget (`tests/test_io_video.py`)
- [ ] Decode via FFmpeg for codecs OpenCV handles poorly

## M2 — Reconstruction backends — scaffolded

- [x] COLMAP backend (features → sequential/exhaustive matching → mapper)
- [x] Dense MVS backend (undistort → patch-match → fusion)
- [x] Poisson/Delaunay meshing + texture mapping; Open3D/Trimesh path
- [x] End-to-end validated on the bundled sample clip (20 keyframes → 21
      registered images → 246k dense points → 31,883-vertex mesh)
- [ ] Publish baseline metrics against NTRO reference data
- [ ] Optional pycolmap in-process backend for the `sfm` extra

## M3 — Georeferencing & metric accuracy — scaffolded

- [x] WGS84 ↔ ECEF ↔ ENU conversions, similarity (Umeyama) fit on camera centers
- [x] GPS RMSE reporting, camera-track GeoJSON, georeferenced PLY output
- [ ] IMU-aided orientation priors and barometric fusion
- [ ] RTK/PPK weighting and GCP import for absolute checks

## M4 — AI depth & dynamic-object handling — planned

- [x] Monocular depth backend (Depth Anything V2) producing per-frame maps
- [ ] Scale-calibrated depth fusion to fill occluded surfaces
- [ ] Semantic segmentation of vegetation/roads for class-aware meshing
- [ ] Learned deblurring (NAFNet-style) behind the `ai` extra

## M5 — Near-real-time & delivery — planned

- [ ] Tile-streaming pipeline with incremental SfM for live UAV downlink
- [ ] FastAPI service (`api` extra): submit video, poll status, download model
- [ ] 3D Tiles / glTF export for browser visualization
- [ ] GPU acceleration profile

## M6 — Evaluation & hardening — planned

- [ ] Benchmark harness: accuracy, completeness, scale error, latency
- [ ] Golden datasets and regression thresholds in CI (`slow` marker)
- [ ] Field trial with the NTRO-provided real-time dataset
