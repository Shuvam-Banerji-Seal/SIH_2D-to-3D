# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Fast profile** (`configs/fast.yaml`), the deliverable: one NVDEC decode, passes and
  overlap-band keyframes with an "is it 3D?" verdict, structure from motion from RAFT flow tracks
  (pycolmap global mapper), dense depth from flow triangulation completed by Depth Anything V2,
  GPU TSDF fusion and GPU texture baking.
- Exports in every format the problem statement asks for: OBJ, PLY, LAS, GeoTIFF (DSM and
  orthophoto), glTF/GLB and FBX, plus a self-contained three.js viewer (`drone3d view`).
- **Accurate profile** (`configs/accurate.yaml`): spirula-studio GPU structure from motion and
  depth-supervised 3D Gaussian Splatting with a Marigold v2 depth prior, meshes from the splats and
  gsplat fly-throughs.
- Ground-levelled 4-DoF georeferencing that survives a straight single-pass track.
- A resident GPU engine (`drone3d engine`), the web console (`drone3d ui`) and live reconstruction
  from a stream, segment by segment.
- `drone3d generate` and `drone3d complete`: a generated, completed model of a run's subject.
- `experiments/`, `paper/` and `promo/`: the scripts that produce every number in the paper, the
  paper itself and the explainer film.
- `drone3d.depth.assets`: one torch-free resolver for the Marigold weights directory.

### Changed

- `drone3d doctor` checks every core dependency and the extras the code actually imports, and exits
  non-zero when a core dependency is missing. It no longer crashes on an install without `torch`.
- The engine's capability probe honours `$DEPTH_ASSETS_DIR` like the Marigold loader does.
- `drone3d view` serves its extra MIME types from a handler subclass instead of mutating the
  standard library's shared table.
- `scipy`, `pillow` and `tifffile` are declared as core dependencies and `scikit-image` under the
  `mesh` extra: the code imports them directly but they were only installed as a side effect of the
  `depth` extra. `httpx2` joins the dev group for the API tests.
- Ruff targets Python 3.12, the project's minimum, and `src` and `tests` are formatted (`config.py` is
  exempt: the console reads each option's help text from the comment on its source line).
- The README and paper generators skip a throughput run in which jobs failed.
- Tests that need PyTorch, Open3D, trimesh or laspy skip when it is absent, so the suite runs on a
  core-and-dev install.
- `CONTRIBUTING.md`, `tests/README.md` and the four remaining `plans/` documents describe the current
  code base; the README's measured table is filled in from `paper/figures/all_maps.json`.

### Removed

- The `docs` dependency group (there is no MkDocs site), the unused `slow`/`gpu` pytest markers, the
  unused `pdoom-video` submodule and the one-off `tools/record_ui/gl_probe.mjs`.
- Four files of `plans/` (a superseded strategy record, an evolution log and spec proposals for the
  AI agent's own operating rules, a session hand-off); recoverable from git history.

## [0.1.0] - 2026-09-20

### Added

- uv-first project scaffold (`pyproject.toml`, PEP 735 dependency groups, committed `uv.lock`,
  Makefile targets, pre-commit hooks, CI workflow).
- Typed pipeline configuration with YAML profiles and dotted `--set` overrides.
- `drone3d` CLI: `run` / `reconstruct`, `init-config`, `doctor`, `version`.
- Ingest with telemetry parsers for CSV, DJI SRT, GPX and JSON, with interpolation.
- Georeferencing: WGS84 ↔ ECEF ↔ ENU conversions, robust similarity fit on camera centres, GPS
  RMSE, georeferenced PLY and camera-track GeoJSON.
- Metrics: cloud bounds, voxel coverage and density, Chamfer distance, completeness and scale
  error; standalone HTML run report with contact sheet.
- Documentation: problem statement mapping, architecture, data formats, roadmap, contributing guide.

The scaffold's first reconstruction code (frame preprocessing, COLMAP command-line SfM/MVS backends,
Poisson meshing) was replaced on 2026-09-27 by the GPU pipeline above.

[Unreleased]: https://github.com/Shuvam-Banerji-Seal/SIH_2D-to-3D/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Shuvam-Banerji-Seal/SIH_2D-to-3D/releases/tag/v0.1.0
