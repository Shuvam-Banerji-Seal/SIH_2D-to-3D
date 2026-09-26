# 00 — Understanding

Read sweep: 2026-09-26. All repo files read except `LICENSE` (verbatim GPL-2.0),
`uv.lock` (lockfile, header inspected), and the binary `.webm` (probed via ffprobe/OpenCV).

## Restated problem

`drone3d` = CLI + library that turns **one single-pass drone video** (+ optional
GPS/IMU log) into a **georeferenced, metrically-accurate, textured 3D model**.
Smart India Hackathon PS **26158** (NTRO). Deliverables: sparse/dense cloud,
textured mesh, georeferenced PLY + camera-track GeoJSON, metrics, one-page HTML report.

## What the repo actually is

| Aspect | State (verified this session) |
| --- | --- |
| Maturity | `0.1.0`, 3 commits, "Early scaffold" per README |
| Structure | `src/drone3d` package, 8 pipeline stages, artifact-based contracts |
| Tests | **None.** `tests/` holds only a `README.md` (planned layout) |
| CI | ruff lint+format, import smoke, `drone3d doctor`, config-load check; pytest guarded by `ls tests/test_*.py` |
| Env on this machine | `uv` ✓, py3.10.12 ✓, ffmpeg/ffprobe ✓, **COLMAP ✗**, **no `.venv`** (package not installed) |
| Sample data | 1 × `.webm` — 2560×1440 VP9, 23.976 fps, **457 frames / 19.06 s**, 22.7 MB |
| Telemetry for it | **None present** (no `.csv/.srt/.gpx/.json` in `datasets/`) |

## Pipeline (stage → artifact contract)

| # | Stage | Reads | Writes | Backend |
| --- | --- | --- | --- | --- |
| 1 | `ingest` | `ingest.video`, `ingest.telemetry` | `frames/`, `ingest/frames.csv`, `video_info.json`, `telemetry_summary.json` | OpenCV |
| 2 | `preprocess` | `ingest/frames.csv` | `frames_selected/` (+`masks/`), `preprocess/selected_frames.csv`, `selection_summary.json` | OpenCV |
| 3 | `sfm` | images dir | `sfm/database.db`, `sparse/`, `sparse.ply`, `sparse_txt/`, `result.json` | COLMAP CLI |
| 4 | `dense` | sfm `result.json` + images | `dense/dense/fused.ply` or `dense/depth/*.npy` | COLMAP MVS / HF depth |
| 5 | `mesh` | `dense.result.json.fused_ply` | `mesh/mesh-*.ply`, `mesh/textured/mesh.obj` | COLMAP poisson/delaunay/texture, Open3D |
| 6 | `georef` | sfm text model + `ingest/frames.csv` GPS | `georef/georeferenced_*.ply`, `camera_track.geojson`, `result.json` | NumPy (Umeyama) |
| 7 | `metrics` | all prior results | `metrics/metrics.json` | NumPy |
| 8 | `report` | everything | `report.html`, then `manifest.json` (written by `Pipeline.run`) | OpenCV |

Graceful degradation: every backend missing → stage `skipped`, run continues,
`PipelineResult.ok` still true if all are `ok|skipped`.

## Module inventory (line counts)

`pipeline.py` 668 · `io/telemetry.py` 417 · `types.py` 253 · `config.py` 276 ·
`io/video.py` 287 · `report/html.py` 265 · `cli.py` 184 · `utils/ply.py` 177 ·
`preprocess/quality.py` 170 · `metrics/quality.py` 162 · `preprocess/stabilize.py` 143 ·
`mesh/texturing.py` 138 · `sfm/features.py` 135 · `geo/georef.py` 130 ·
`mesh/colmap_mesher.py` 118 · `dense/mono_depth.py` 112 · `sfm/colmap_model.py` 112 ·
`preprocess/dynamic.py` 106 · `dense/mvs.py` 104 · `geo/enu.py` 98 ·
`geo/projection.py` 92 · `preprocess/deblur.py` 90 · `sfm/colmap_backend.py` 194 ·
plus `base.py` factories, `features.py`, `exceptions.py`, `logging_utils.py`, `utils/shell.py`.

## Key algorithms (as implemented)

- **Keyframe selection** — composite score `0.6·sharpness + 0.25·exposure + 0.15·contrast`,
  greedy accept with `min_spacing_s` enforced via `bisect`+`insort`; oversample ×3 by default.
- **Georef** — COLMAP camera centers ↔ per-frame GPS in local ENU (WGS84→ECEF→ENU, pure NumPy),
  **Umeyama** closed-form similarity (`solve_similarity`), reports `gps_rmse` horiz/vert/3D.
- **Metrics** — bounds, voxel coverage, symmetric Chamfer (chunked brute force, ≤50k samples),
  completeness @0.5 m, geometric scale error.
- **Telemetry** — alias-table column auto-detect; parsers for CSV/TSV, DJI SRT, GPX, JSON;
  `bisect` linear interpolation onto frame timestamps.
- **PLY** — hand-rolled ASCII + binary-LE reader/writer, no Open3D dependency.

## Findings from the read (each verified, not assumed)

| ID | Finding | Evidence | Severity |
| --- | --- | --- | --- |
| F1 | `TelemetrySample.to_dict()` raises `AttributeError: no '__dict__'` — class is `@dataclass(slots=True)` but method does `self.__dict__` | executed: `PYTHONPATH=src python3 -c ...` → failed | **High (latent)** — no caller inside `src/` today, but it is exported public API |
| F2 | Dynamic masks are **written but never consumed**: `preprocess` writes `frames_selected/masks/*.png`; `ColmapSfMBackend._extract_features` never passes `--ImageReader.mask_path` and no code reads `masks/` | `grep -rn mask src` → only writers; `docs/data-formats.md` claims "COLMAP mask input" | **High** — PS criterion 5 (no dynamic-object ghosting) is currently unmet by the wiring |
| F3 | `_largest_model` boolean-precedence bug: `path.is_dir() and (…images.bin).exists() or (…images.txt).exists()` parses as `(A and B) or C` | source inspection + precedence demo printed | Low (latent; in practice `iterdir()` entries are dirs) |
| F4 | `extract_frames` ignores **both** `sample_fps` and `max_frames` when `CAP_PROP_FRAME_COUNT <= 0`: `wanted=None` ⇒ every frame written, no early break | code path traced; **not triggered by sample video** (frame_count=457) | Medium — will bite on streams/unknown-length containers |
| F5 | `point_to_cloud_distances` allocates `chunk × N_ref × 3` float64 ⇒ with defaults `1000 × 50 000 × 3 × 8 B ≈ 1.2 GB` per chunk, and 2.5e9 pair distances total | arithmetic from `_CHUNK=1000`, `_MAX_SAMPLES=50_000` | Medium — OOM/slowness risk on reference-cloud runs |
| F6 | `mono_depth.depth()` docstring says "1 = farthest"; Depth-Anything raw output is inverse-depth (larger = nearer), and min-max normalisation preserves that polarity | code read; **[HYPOTHESIS]** — not yet verified against model card | Medium — would silently invert any future depth fusion |
| F7 | No telemetry file exists for the shipped dataset ⇒ `georef` will always `skip` on this data as delivered | `ls datasets/` → 1 file | Blocker for the georef/metrics criteria on sample data |
| F8 | COLMAP absent on this machine ⇒ `sfm`/`dense`/`mesh` all `skip` with `allow_missing: true` | `which colmap` → not found | Blocker for end-to-end validation |
| F9 | `Makefile` `run` target calls `drone3d reconstruct` — valid (alias of `run`), not a bug | `cli.py` `aliases=["reconstruct"]` | Info |
| F10 | Repo `.gitignore` forbids committing footage, yet `datasets/*.webm` (22.7 MB) **is** committed (commit `788a9cf`) | `git log`, `CONTRIBUTING.md` "Data hygiene" | Info — intentional sample, contradicts contributing guide |

## Success criteria (from PS + repo docs)

| # | Criterion | Target | Verifiable here? |
| --- | --- | --- | --- |
| 1 | Geometric accuracy | ≤1× GSD horiz, ≤2× vert | Needs reference/GPS checkpoints |
| 2 | Scale correctness | ≤2 % w/o GCPs | Needs `expected_extent_m` or reference |
| 3 | Completeness | ≥80 % @0.5 m | Needs reference cloud |
| 5 | No dynamic ghosting | visible check | Needs F2 fixed + run |
| 6 | Georeferencing | ≤3 m horizontal | Needs telemetry (F7) |
| 7 | Robustness | no crash, graceful degradation | **Testable now** |
| 9 | Usability | one command + `report.html` | **Testable now** (minus COLMAP) |
| 10 | Reproducibility | same input+config → same metrics | **Testable now** |

## Constraints

- GPL-2.0-only. Optional heavy deps behind extras: `ai`, `sfm`, `mesh`, `geo`, `api`.
- Ruff: line length 100, `E,F,I,UP,B,SIM,C4,RET,PTH`, `E501` ignored.
- Pytest markers `slow`, `gpu` declared but no tests exist to use them.
- No network assumptions in tests (per `tests/README.md`).

## Open questions

| ID | Question | Prio |
| --- | --- | --- |
| Q1 | Should masks be wired via `--ImageReader.mask_path`, or is a sibling `masks/` dir the intended convention? (needs COLMAP docs check) | High |
| Q2 | Is there NTRO telemetry arriving separately, or must we synthesise GPS for the sample video? | High |
| Q3 | Install COLMAP (`apt`) vs. accept `skipped` stages for validation? | High |
| Q4 | Does `uv sync` resolve offline here (is there network)? | Medium |
