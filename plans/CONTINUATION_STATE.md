# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 4 |
| Phase | AUDIT (cycle 4) |
| What I did | Completed priorities 1–4 (type errors, F7 telemetry, roadmap, commit `87fa8ab`). Fixed **F16** (`write_ply` colour-block data corruption) with 8 tests. Ran the **double-audit: two consecutive green passes**. Verified criterion 10 (metrics byte-identical across runs). |
| What worked | 50 tests green × 2 passes, ruff clean, format clean, pipeline 8/8 stages, metrics reproducible byte-for-byte |
| What failed | Nothing this cycle |
| Errors remaining | **F7 is a data gap**, not a code defect: the bundled sample video carries no GPS. Criteria 1, 2, 3, 6 cannot be scored without a genuine flight log or reference cloud. |
| Next priorities | 1) Add `tests/test_telemetry.py` (CSV/SRT/GPX/JSON parsing — the largest remaining coverage gap) · 2) Add `tests/test_preprocess.py` + `tests/test_config.py` · 3) Obtain real telemetry to score criteria 2 & 6 · 4) Delete nothing; commit incremental test additions |
| Blockers | External data only: needs a real drone flight log (or NTRO reference cloud) to validate accuracy criteria |
| Audit status | **DOUBLE_PASS** (two consecutive green verification waves, see `05-audit-log.md`) |

## IMPORTANT: prior priorities are DONE — do not redo them

The harness prompt has been repeating these stale items. All verified complete on disk:

| Old priority | Status | Evidence |
|---|---|---|
| 1) Fix 2 pre-existing type errors | **DONE** | `colmap_backend.py:183` `class _ModelStats(TypedDict)`; `pipeline.py:472` `located = [...]` narrowing. LSP clean. |
| 2) F7/Q2 telemetry | **DONE (synthesised)** | `tools/synthesize_telemetry.py`; `georef` runs: 19 tie points, horizontal RMSE 0.00 m |
| 3) `docs/roadmap.md` stale | **DONE** | M1/M2 checkmarks + real numbers ("End-to-end validated on the bundled sample clip") |
| 4) Commit the work | **DONE** | `87fa8ab` — 14 modified + 8 new files; worktree clean |

## Verified state

| Check | Result |
|---|---|
| `uv run pytest` | **50 passed** (2 consecutive clean passes, `-p no:cacheprovider`) |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 66 files already formatted |
| Pipeline | ingest/preprocess/sfm/dense/mesh/georef/metrics/report = **8/8 OK** |
| Reproducibility (criterion 10) | `metrics.json` **byte-identical** across re-runs |
| COLMAP mask polarity | `0 = ignore` **[VERIFIED: colmap.github.io/faq.html]** — "no features will be extracted in regions where the mask image is black (pixel intensity value 0 in grayscale)" |
| Depth polarity (F6) | `1 = nearest` **[VERIFIED: empirical]** — real photo: near 0.627 vs far 0.308 |
| Nothing deleted (A9) | Confirmed; no removals |

## Defect ledger (all fixed except F7)

| ID | Defect | Status |
|---|---|---|
| F1 | `TelemetrySample.to_dict` slots crash | fixed |
| F2 | Masks written but never fed to COLMAP (3-way: location, name, polarity) | fixed |
| F3 | `_largest_model` boolean precedence | fixed |
| F4 | Frame budget ignored when `frame_count<=0` | fixed |
| F5 | Chamfer ~2.4 GB/chunk | fixed |
| F6 | Depth polarity docstring wrong | fixed |
| **F7** | **Sample video has no GPS** | **DATA GAP — needs real flight log** |
| F8 | COLMAP absent | fixed (`.tools/colmap-env`, 4.2.0 CUDA) |
| F12 | `.tools/` broke ruff + 4.4G untracked | fixed |
| F13 | `model_converter` TXT aborted on missing dir | fixed |
| F14 | `poisson_mesher` SIGSEGV | fixed (Delaunay fallback) |
| F15 | `delaunay_mesher --output_type` rejected | fixed |
| **F16** | **`write_ply` colour-block data corruption** | fixed + 8 tests |

## Test suite: 50 tests / 7 files

| File | Tests | Covers |
|---|---|---|
| `test_geo.py` | 15 | ECEF/ENU round-trips, Umeyama, GPS RMSE |
| `test_sfm_colmap_backend.py` | 8 | COLMAP argv, mask wiring, model selection, F13 |
| `test_ply.py` | 8 | **F16** interleaved binary round-trip |
| `test_io_video.py` | 6 | Frame sampling, `max_frames` (mutation-verified) |
| `test_metrics_quality.py` | 5 | Chamfer vs brute force, memory clamping |
| `test_mesh_colmap_mesher.py` | 4 | Poisson→Delaunay fallback, F15 |
| `test_types.py` | 4 | F1 slots `to_dict` |

## File Manifest

| File | Status | Last modified |
|------|--------|---------------|
| plans/00-understanding.md | stale (F-table superseded by ledger above) | cycle 1 |
| plans/01-research.md | current | cycle 2 |
| plans/02-strategy.md | current | cycle 2 |
| plans/04-decisions.md | stale (pre-dates F13–F16) | cycle 2 |
| plans/05-audit-log.md | **current** | cycle 4 |
| plans/CONTINUATION_STATE.md | **current** | cycle 4 |
| plans/INFINITY_DONE | absent — accuracy criteria blocked on external data | - |

## Continuation Prompt Hints

**Do not redo priorities 1–4** (see table above). Do not re-research Spirula (D1 settled).
Next highest leverage:
1. `tests/test_telemetry.py` — CSV/SRT/GPX/JSON parsing + interpolation is the biggest
   uncovered module (`io/telemetry.py`, 417 lines, zero tests).
2. `tests/test_config.py` (load/`--set` overrides) and `tests/test_preprocess.py`
   (quality metrics, keyframe selection).
3. Real telemetry / reference cloud to turn criteria 2 & 6 from blocked into scored.
4. Commit incremental test additions. Keep everything green (`uv run pytest`, `ruff`).
