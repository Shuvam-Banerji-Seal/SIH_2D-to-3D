# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 6 |
| Phase | TEST → AUDIT (cycle 6) |
| What I did | Closed every testable coverage gap and found **2 more real defects (F18, F19)** while doing so. Test suite **50 → 205 tests** (14 files). Re-ran the double-audit: **two consecutive green waves**, 74% coverage. |
| What worked | Coverage-driven targeting found F19 (a real crash affecting *all* feature methods on OpenCV 5.x) that code reading had missed entirely |
| What failed | 10 test-side bugs of my own (API guesses); all diagnosed and corrected, documented in commit messages |
| Errors remaining | **F7 only — an external data gap**, not code. Criteria 1, 2, 3, 6 need a genuine flight log or reference cloud. |
| Next priorities | **None actionable in code.** See "What is actually left". |
| Blockers | External data (real telemetry / NTRO reference cloud) |
| Audit status | **DOUBLE_PASS** (waves A & B, cycle 6 — 205 tests each, ruff clean, format clean) |

## Status: all in-repo work is discharged

| Item | Status |
|---|---|
| Spirula evaluation (D1) | settled — do not add; parked as optional `splat` stage (D3) |
| F1–F8, F12–F16, F18, F19 | **all fixed and tested** |
| F7 | **external data gap** — needs a real flight log |
| Test suite | 205 tests / 14 files / 74% coverage |
| Lint + format | clean |
| Double-audit | **PASS** (2 consecutive waves) |
| Reproducibility (criterion 10) | verified 3× — metrics byte-identical |
| Commits | 8 (`87fa8ab` … `613303d`) |

## Final defect ledger

| ID | Defect | Found by |
|---|---|---|
| F1 | `TelemetrySample.to_dict` slots crash | code read |
| F2 | Masks never fed to COLMAP (location + name + **polarity**) | code read + live COLMAP docs |
| F3 | `_largest_model` precedence | code read |
| F4 | Frame budget ignored when `frame_count<=0` | code read |
| F5 | chamfer ~2.4 GB/chunk | arithmetic |
| F6 | Depth polarity docstring wrong | empirical run |
| F7 | **Sample video has no GPS** | ffprobe |
| F8 | COLMAP absent | `doctor` |
| F12 | `.tools/` broke ruff | own regression |
| F13 | `model_converter` TXT missing dir | running pipeline |
| F14 | `poisson_mesher` SIGSEGV | running pipeline |
| F15 | `delaunay_mesher --output_type` rejected | running pipeline |
| F16 | `write_ply` colour-block **data corruption** | running pipeline |
| F18 | CSV no-header guard dead code | writing tests |
| F19 | Eager detector map broke all feature methods on OpenCV 5.x | writing tests |

## Test suite

| File | Tests | Covers |
|---|---|---|
| `test_geo.py` | 15 | ECEF/ENU, Umeyama, GPS RMSE |
| `test_telemetry.py` | 33 | CSV/SRT/GPX/JSON, interpolation |
| `test_config.py` | 20 | YAML + `--set` coercion |
| `test_features.py` | 16 | SIFT/ORB detect, matching, frame graph |
| `test_metrics_quality.py` | 15 | chamfer, bounds, voxel, completeness |
| `test_colmap_model.py` | 14 | images.txt/points3D parsing |
| `test_preprocess_filters.py` | 14 | stabilize, deblur, masking |
| `test_preprocess.py` | 17 | quality, keyframe selection |
| `test_projection.py` | 17 | intrinsics, GSD, tangent plane |
| `test_ply.py` | 8 | F16 interleaved round-trip |
| `test_sfm_colmap_backend.py` | 8 | COLMAP argv, masks, F13 |
| `test_io_video.py` | 6 | frame sampling (mutation-verified) |
| `test_mesh_colmap_mesher.py` | 4 | F14/F15 fallback |
| `test_types.py` | 4 | F1 slots |
| `test_pipeline_smoke.py` | 4 | end-to-end + criterion 7 |
| `test_cli.py` | 10 | CLI entry points |

## What is actually left (nothing blocking)

1. **Real flight log / NTRO reference cloud** to turn criteria 1, 2, 3, 6 from
   blocked into scored — external data, not code.
2. Optional: `slow`/`gpu`-marked tests for `dense/mvs.py` and `mesh/texturing.py`
   (need COLMAP/Open3D at runtime; excluded from the fast suite by design).
3. Optional: publish baseline metrics against NTRO reference data (roadmap M2).

## File Manifest

| File | Status | Last modified |
|------|--------|---------------|
| plans/00-understanding.md | stale (superseded by ledger above) | cycle 1 |
| plans/01-research.md | current | cycle 2 |
| plans/02-strategy.md | current | cycle 2 |
| plans/04-decisions.md | stale (pre-dates F13–F19) | cycle 2 |
| plans/05-audit-log.md | **current** | cycle 6 |
| plans/CONTINUATION_STATE.md | **current** | cycle 6 |
| plans/INFINITY_DONE | absent — accuracy criteria blocked on external data | - |

## Continuation Prompt Hints

All in-repo work is discharged and the double-audit has **passed twice**. Do
**not** re-research Spirula (D1) or re-diagnose any F-item (all fixed).

If continuing, the only work left is:
1. Integrate real telemetry when supplied — rerun `georef`, compare `gps_accuracy`
   in `outputs/*/georef/result.json` against the ≤3 m criterion.
2. Optionally add `slow`/`gpu` tests for `dense/mvs.py` / `mesh/texturing.py`.
3. **`plans/INFINITY_DONE` should be written only once criteria 1/2/3/6 are
   scored against real reference data** — until then the accuracy targets are
   unverified and the loop should continue.
