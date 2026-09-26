# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 3 |
| Phase | IMPLEMENT → TEST (cycle 3) |
| What I did | Closed the interrupted venv verification (all 13 extras import; slow but healthy). Installed COLMAP 4.2.0 CUDA into `.tools/colmap-env` (dedicated env, NOT miniforge base — would break the uv venv). Fixed **F1, F2, F3, F4, F5, F6, F8** and found + fixed **3 new defects (F13/F14/F15)** by running the real pipeline. Wrote the repo's **first tests: 27 passing**. Full pipeline now runs end-to-end. |
| What worked | Mutation-checked F4 test (red-then-green). Real pipeline run exposed 3 defects no amount of reading would have found. Verifying COLMAP masks against live docs caught an **inverted-polarity** bug that would have silently kept the vehicles. |
| What failed | 2 wrong hypotheses, both retracted: (a) "no NVIDIA Vulkan" (R2-a), (b) "TXT export fails because of COLMAP 4 rig/frames.bin" (R3-a) — real cause was a missing `mkdir`. Also my synthetic 2-tone depth test gave a **wrong** verdict; a real photo corrected it (R4-a). |
| Errors remaining | **F7** (data gap: sample video has no GPS — not fixable in code). Pre-existing LSP type errors: `colmap_backend.py:122-123` (`_read_model_stats` returns `float\|int`), `pipeline.py:471` (`to_local` gets `float\|None`) |
| Next priorities | 1) Fix the 2 pre-existing type errors · 2) F7/Q2: obtain or synthesise telemetry to unblock `georef` + criteria 2/6 · 3) `docs/roadmap.md` still says "no tests" — stale · 4) Commit the work |
| Blockers | F7 needs telemetry data from the user (real drone log) — nothing to fix in code |
| Audit status | PARTIAL — macro-audit checklist drafted in `05-audit-log.md`, not yet a double-pass |

## What is now working (verified by execution)

| Stage | Status | Evidence |
|---|---|---|
| ingest | OK | 20 frames @ 1 fps from 19.1 s |
| preprocess | OK | 20/20 selected, **20 masks** in `preprocess/masks/frame_0000NN.jpg.png` |
| sfm | OK | 21 images, 1286 points (COLMAP CUDA) |
| dense | OK | 246,354 points |
| mesh | OK | 31,883 vertices / 63,904 faces (Poisson→Delaunay fallback) |
| georef | SKIPPED | "no frame GPS fixes (ingest.telemetry missing?)" — graceful, expected |
| metrics / report | OK | 2 clouds summarised, `report.html` written |

Command (needs COLMAP on PATH):
```bash
export PATH="$PWD/.tools/colmap-env/bin:$PATH"
uv run drone3d run --config configs/fast.yaml --run-dir outputs/sample_fast \
  --set "ingest.video=datasets/Aerial Views of Rural Riches： Drone Shot of Farmland 🌾🚁 [p8eRmxosalI].webm" \
  --set preprocess.dynamic_masking=true
```

## Defect ledger

| ID | Defect | Status | Fix |
|---|---|---|---|
| F1 | `TelemetrySample.to_dict` crashes (`slots` has no `__dict__`) | **fixed** | `dataclasses.fields()` |
| F2 | Masks written but never fed to COLMAP | **fixed** | `--ImageReader.mask_path`; moved out of images dir; **renamed** `<img>.png`; **inverted** polarity (0 = ignore) |
| F3 | `_largest_model` boolean precedence | **fixed** | parentheses |
| F4 | `max_frames`/`sample_fps` ignored when `frame_count <= 0` | **fixed** | stream with `step` + hard budget |
| F5 | chamfer allocated ~2.4 GB/chunk | **fixed** | BLAS identity + 64 MB budget |
| F6 | mono_depth docstring polarity wrong | **fixed** | "1 = nearest", verified empirically |
| F7 | Sample video has **no GPS** | **data gap** | needs telemetry from user |
| F8 | COLMAP absent | **fixed** | `.tools/colmap-env` (4.2.0 CUDA) |
| F13 | `model_converter --output_type TXT` aborted: output dir missing | **fixed** | `text_dir.mkdir()` |
| F14 | `poisson_mesher` SIGSEGV in Poisson trimmer | **fixed** | logged fallback → delaunay |
| F15 | `delaunay_mesher --output_type` unrecognised (never worked) | **fixed** | drop the flag |
| F12 | `.tools/` broke ruff (1011 errors) + untracked 4.4 GB | **fixed** | `extend-exclude` + `.gitignore` |

## Test suite (was 0, now 27)

| File | Tests | Covers |
|---|---|---|
| `tests/test_types.py` | 4 | F1 |
| `tests/test_io_video.py` | 6 | F4 (mutation-verified) |
| `tests/test_sfm_colmap_backend.py` | 8 | F2, F3, F13 |
| `tests/test_mesh_colmap_mesher.py` | 4 | F14, F15 |
| `tests/test_metrics_quality.py` | 5 | F5 |

## File Manifest

| File | Status | Last modified |
|------|--------|---------------|
| plans/00-understanding.md | stale (F-table now superseded) | cycle 1 |
| plans/01-research.md | current | cycle 2 |
| plans/02-strategy.md | current | cycle 2 |
| plans/04-decisions.md | stale (needs F13-F15) | cycle 2 |
| plans/05-audit-log.md | **new** | cycle 3 |
| plans/CONTINUATION_STATE.md | **current** | cycle 3 |
| plans/INFINITY_DONE | absent (correct) | - |

## Continuation Prompt Hints

Do **not** re-research Spirula (D1) or re-diagnose F1–F8/F13–F15 (all fixed and tested).
Highest leverage next:
1. Fix the 2 pre-existing type errors (`colmap_backend._read_model_stats` int typing; `pipeline.to_local` optional handling) — cheap, removes LSP noise.
2. **F7 is the real remaining blocker** for scored criteria 2 & 6: the shipped sample has no GPS (verified via ffprobe — only `language`/`DURATION`/`ENCODER` tags). Either obtain a real drone log, or synthesise a clearly-labelled synthetic track so `georef`/`metrics` can be exercised and tested.
3. Update `docs/roadmap.md` (claims no tests) and `tests/README.md` (now has real tests).
4. Commit: 8 modified + 4 new test files + `plans/`, nothing deleted.
