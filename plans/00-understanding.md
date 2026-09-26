# 00 — Understanding

Read sweep: 2026-09-26. Reconciled at cycle 9 (this file was stale — its Findings
table described the codebase *before* 15 defects were fixed). Current state lives
in `CONTINUATION_STATE.md` and `08-final-verification-report.md`; this file keeps
the durable understanding of the project itself.

## Restated problem

`drone3d` = CLI + library that turns **one single-pass drone video** (+ optional
GPS/IMU log) into a **georeferenced, metrically-accurate, textured 3D model**.
Smart India Hackathon PS **26158** (NTRO). Deliverables: sparse/dense cloud,
textured mesh, georeferenced PLY + camera-track GeoJSON, metrics, one-page HTML report.

## What the repo is

| Aspect | State |
| --- | --- |
| Maturity | `0.1.0`, 18 commits |
| Structure | `src/drone3d`, 8 artifact-contract stages |
| Tests | **271 passing** across 20 files; 81 % coverage |
| CI | ruff lint+format, import smoke, `drone3d doctor`, config load, pytest |
| Env | `uv` `.venv` (CPython 3.12.8, all extras), COLMAP 4.2.0 CUDA in `.tools/colmap-env` |
| Sample data | 1 × `.webm` — 2560×1440 VP9, 23.976 fps, 457 frames / 19.06 s, 22.7 MB |
| Telemetry for it | **None** — this is the one remaining blocker |

## Pipeline (stage → artifact contract)

| # | Stage | Reads | Writes | Backend |
| --- | --- | --- | --- | --- |
| 1 | `ingest` | `ingest.video`, `ingest.telemetry` | `frames/`, `ingest/frames.csv`, `video_info.json` | OpenCV |
| 2 | `preprocess` | `ingest/frames.csv` | `frames_selected/`, `preprocess/masks/`, `selected_frames.csv` | OpenCV |
| 3 | `sfm` | images dir | `sfm/database.db`, `sparse/`, `sparse.ply`, `sparse_txt/` | COLMAP CLI |
| 4 | `dense` | `sfm` result + images | `dense/dense/fused.ply` or `dense/depth/*.npy` | COLMAP MVS / HF depth |
| 5 | `mesh` | `dense` result | `mesh/mesh-*.ply`, `mesh/textured/` | COLMAP poisson/**delaunay**/texture, Open3D |
| 6 | `georef` | `sfm` text model + frame GPS | `georef/georeferenced_*.ply`, `camera_track.geojson` | NumPy (Umeyama) |
| 7 | `metrics` | all results | `metrics/metrics.json` | NumPy |
| 8 | `report` | everything | `report.html`, `manifest.json` | OpenCV |

Graceful degradation is a design pillar: a missing backend ⇒ stage `skipped`, run
continues, `PipelineResult.ok` stays true.

## Artifact contracts worth knowing

- **Dynamic masks** live at `preprocess/masks/<image-name>.png` — *outside* the
  images dir (COLMAP scans `image_path` recursively) — named `<image>.png` with
  **0 = ignore** polarity, consumed via `--ImageReader.mask_path`.
- **`model_converter --output_type TXT`** requires its output directory to
  already exist; `--output_type PLY` does not.
- **PLY binary is interleaved per vertex.** Writing positions then colours as
  separate blocks silently corrupts every vertex past the first.
- **`delaunay_mesher`** has no `--output_type` flag in COLMAP 4.x.
- **OpenCV 5.x removed AKAZE** and its stubs omit `SIFT_create` /
  `VideoWriter_fourcc` (both work at runtime).

## Key algorithms

- **Keyframe selection** — `0.6·sharpness + 0.25·exposure + 0.15·contrast`,
  greedy accept with `min_spacing_s` enforced via `bisect`/`insort`.
- **Georef** — camera centres ↔ per-frame GPS in local ENU (WGS84→ECEF→ENU,
  pure NumPy), **Umeyama** closed-form similarity, reports GPS RMSE.
- **Metrics** — bounds, voxel coverage, symmetric Chamfer (BLAS identity
  `|a−b|²=|a|²+|b|²−2a·b`, 64 MB budget), completeness, scale error.
- **Telemetry** — alias-table column detection; CSV/TSV, DJI SRT, GPX, JSON;
  `bisect` linear interpolation onto frame timestamps.

## Success criteria status

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criterion 1
has supporting evidence (SfM reprojection error 0.30 px ≈ 0.3× GSD). Criteria
**2, 3, 6 are blocked on external data** — see `08-final-verification-report.md`.

## Constraints

- GPL-2.0-only. Optional extras: `ai`, `sfm`, `mesh`, `geo`, `api`.
- Ruff line length 100; `E,F,I,UP,B,SIM,C4,RET,PTH`, `E501` ignored.
- Tests must be hermetic: stub `run_command`/`which`, assert on argv.

## Open questions

| ID | Question | Prio |
| --- | --- | --- |
| Q6 | Camera model / FoV of the sample clip — needed for a real GSD figure | High |
| Q7 | Real flight log or NTRO reference cloud — unblocks criteria 2, 3, 6 | **High** |
