# 00 — Understanding

Originally written cycle 1 as the read sweep of the repository. **Reconciled at
cycle 9** — the Findings table is superseded by the defect ledger below; the
rest of the analysis still holds.

## Restated problem

`drone3d` = CLI + library that turns **one single-pass drone video** (+ optional
GPS/IMU log) into a **georeferenced, metrically-accurate, textured 3D model**.
Smart India Hackathon PS **26158** (NTRO). Deliverables: sparse/dense cloud,
textured mesh, georeferenced PLY + camera-track GeoJSON, metrics, one-page HTML report.

## What the repo is now (cycle 9)

| Aspect | State |
| --- | --- |
| Maturity | `0.1.0`, 18 commits, 238 tests, 78% coverage |
| Structure | `src/drone3d`, 8 pipeline stages, artifact-based contracts |
| Env | `.venv` CPython 3.12.8 (`--all-extras`); COLMAP 4.2.0 CUDA at `.tools/colmap-env` |
| Sample data | 1 × `.webm` — 2560×1440 VP9, 23.976 fps, 457 frames / 19.06 s |
| Telemetry for it | **None** — verified by ffprobe (only `language`/`DURATION`/`ENCODER` tags) |

## Pipeline (stage → artifact contract)

| # | Stage | Reads | Writes | Backend |
| --- | --- | --- | --- | --- |
| 1 | `ingest` | `ingest.video`, `ingest.telemetry` | `frames/`, `ingest/frames.csv`, `video_info.json` | OpenCV |
| 2 | `preprocess` | `ingest/frames.csv` | `frames_selected/`, `preprocess/masks/`, `selection_summary.json` | OpenCV |
| 3 | `sfm` | images dir | `sfm/database.db`, `sparse/`, `sparse.ply`, `sparse_txt/` | COLMAP CLI |
| 4 | `dense` | sfm `result.json` + images | `dense/dense/fused.ply` | COLMAP MVS / HF depth |
| 5 | `mesh` | `dense/.../fused.ply` | `mesh/mesh-*.ply`, `mesh/textured/` | COLMAP poisson→**delaunay fallback**, Open3D |
| 6 | `georef` | `sparse_txt/*.txt` + `ingest/frames.csv` | `georef/georeferenced_*.ply`, `camera_track.geojson` | NumPy (Umeyama) |
| 7 | `metrics` | all prior results | `metrics/metrics.json` | NumPy |
| 8 | `report` | everything | `report.html`, `manifest.json` | OpenCV |

Graceful degradation holds: missing backend ⇒ stage `skipped`, run continues.

## Key algorithms

- **Keyframe selection** — composite score `0.6·sharpness + 0.25·exposure + 0.15·contrast`,
  greedy accept with `min_spacing_s` enforced via `bisect`+`insort`.
- **Georef** — GPS↔camera-centre correspondences in local ENU, **Umeyama** closed-form
  similarity, reports `gps_accuracy` RMSE (horiz/vert/3D).
- **Metrics** — bounds, voxel coverage, symmetric Chamfer (BLAS identity, 64 MB budget),
  completeness @0.5 m.
- **PLY** — hand-rolled binary **and ASCII** reader/writer, no Open3D dependency.

## Defects: 15 found and fixed

| ID | Defect | Found by | Status |
| --- | --- | --- | --- |
| F1 | `TelemetrySample.to_dict` slots crash | code read | fixed |
| F2 | Masks never fed to COLMAP (location + name + **polarity**) | code read + live COLMAP docs | fixed |
| F3 | `_largest_model` boolean precedence | code read | fixed |
| F4 | Frame budget ignored when `frame_count<=0` | code read | fixed |
| F5 | chamfer ~2.4 GB/chunk | arithmetic | fixed |
| F6 | Depth polarity documented backwards | empirical model run | fixed |
| **F7** | **Sample video has no GPS** | ffprobe | **DATA GAP** |
| F8 | COLMAP absent | `doctor` | fixed |
| F12 | `.tools/` broke ruff + 4.4 G untracked | own regression | fixed |
| F13 | `model_converter` TXT aborted on missing dir | running the pipeline | fixed |
| F14 | `poisson_mesher` SIGSEGV | running the pipeline | fixed |
| F15 | `delaunay_mesher --output_type` rejected | running the pipeline | fixed |
| F16 | `write_ply` colour-block **data corruption** | running the pipeline | fixed |
| F18 | CSV no-header guard dead code | writing tests | fixed |
| F19 | Eager detector map broke all feature methods on OpenCV 5.x | writing tests | fixed |

**6 of 15 were only findable by executing the pipeline or writing tests.**

## Success criteria (PS) — current status

| # | Criterion | Target | Status |
| --- | --- | --- | --- |
| 1 | Geometric accuracy | ≤1×GSD / ≤2×GSD vert | measured (0.30 px reprojection); needs reference to score |
| 2 | Metric scale | ≤2 % | **blocked: data** |
| 3 | Completeness | ≥80 % @0.5 m | **blocked: data** |
| 4 | Visual quality | recognisable | partial (no human review) |
| 5 | No dynamic ghosting | visual | mechanically verified (F2); visual check pending |
| 6 | Georeferencing | ≤3 m | **blocked: data**; measurement verified |
| 7 | Robustness | no crash | **PASS** |
| 8 | Latency | near-real-time `fast` | measured (see `08-final-verification-report.md`) |
| 9 | Usability | one command + report | **PASS** |
| 10 | Reproducibility | same → same | **PASS** |

## Constraints

- GPL-2.0-only. Heavy deps behind extras: `ai`, `sfm`, `mesh`, `geo`, `api`.
- Ruff line length 100, `E,F,I,UP,B,SIM,C4,RET,PTH`.
- External tooling lives in `.tools/` (git-ignored) and must never be imported by tests.

## Open questions

| ID | Question | Prio | Status |
| --- | --- | --- | --- |
| Q2 | Real telemetry for the sample clip? | High | **answered: none exists** |
| Q4 | Does Spirula write COLMAP-format sparse output? | Low | **[UNKNOWN]**, deferred under D3 |
| Q6 | Reference data for criteria 1, 2, 3, 6? | **High** | **OPEN — the only blocker** |
