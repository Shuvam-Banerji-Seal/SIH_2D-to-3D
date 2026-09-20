# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Unit and smoke tests for ingest, preprocessing, geo and PLY I/O
- Validation of COLMAP SfM/MVS/meshing stages on the NTRO dataset
- FastAPI service and 3D Tiles / glTF delivery
- Benchmark harness with accuracy, completeness and latency thresholds

## [0.1.0] - 2026-09-20

### Added

- uv-first project scaffold (`pyproject.toml`, PEP 735 dependency groups,
  committed `uv.lock`, Makefile targets, pre-commit hooks, CI workflow).
- Typed pipeline configuration with YAML profiles (`fast`, `default`,
  `accurate`) and dotted `--set` overrides.
- `drone3d` CLI: `run` / `reconstruct`, `init-config`, `doctor`, `version`.
- Ingest: video probing, deterministic frame sampling, frame manifests, and
  telemetry parsers for CSV, DJI SRT, GPX and JSON with interpolation.
- Preprocessing: quality scoring (sharpness, exposure, contrast), keyframe
  selection, Wiener/unsharp deblurring, stabilization and dynamic-object masks.
- Reconstruction stage contracts and backends: COLMAP SfM (sequential /
  exhaustive / vocab-tree matching), dense MVS, Poisson/Delaunay meshing with
  texture mapping, and Open3D/Trimesh alternatives.
- Georeferencing: WGS84 ↔ ECEF ↔ ENU conversions, robust similarity fit on
  camera centers, GPS RMSE, georeferenced PLY and camera-track GeoJSON.
- Metrics: cloud bounds/extent, voxel coverage/density, Chamfer distance,
  completeness and scale error; standalone HTML run report with contact sheet.
- Documentation: problem statement mapping (desired outputs and evaluation
  criteria), architecture, data formats, roadmap, contributing guide.

[Unreleased]: https://github.com/Shuvam-Banerji-Seal/SIH_2D-to-3D/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Shuvam-Banerji-Seal/SIH_2D-to-3D/releases/tag/v0.1.0
