# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 22 |
| Phase | AUDIT (cycle 22) |
| What I did | Deliverables re-audit (the F31 axis) found **F32**: PS deliverable 5 names a "scale check" in `metrics.json`, but it was gated on `metrics.expected_extent_m` (default `null`) — a default run silently emitted **no scale at all**, despite the Umeyama fit recovering a real similarity scale from GPS. Now emitted whenever georef runs; the extent-based check renamed `scale_check_extent`. Regression test added. Committed as `ee65708`. |
| What worked | Real-run verification (not just tests): `metrics.json` now carries `scale_check` with `kind=georef_similarity_scale`, `scale=1.0007457985792914`, `relative_error=0.0007457985792913568` |
| What failed | 3× `edit` on stale `oldString` (file already fixed — re-read instead of retrying); one missing `Path` import in my own test (fixed) |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** at 285 (cycle 21) → **re-audit needed at 286** |

## F32 — the scale check was silently absent

`docs/problem-statement.md` deliverable 5: *"Extents and scale error reported in `metrics.json`"*.

| Condition | Before | After |
|---|---|---|
| georef ran, `expected_extent_m` unset (default) | **no `scale_check` key** | `scale_check.kind = georef_similarity_scale` |
| georef ran, `expected_extent_m` set | `scale_check` = extent error | `scale_check_extent` = extent error |
| no georef | absent | absent |

Fix: `src/drone3d/pipeline.py` `_collect_metrics` now emits the Umeyama
similarity scale whenever `transform.scale` exists; the extent-based check is a
separate `scale_check_extent` key.

**Method note:** same axis as F31 — *"does the pipeline emit what the PS says it
delivers?"*. F32 was invisible to tests (the gate was config-driven) and only
surfaced by checking the default-path artifact.

## Defect ledger: 25 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F32** (F9–F11 never existed), plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Checking deliverables against the PS** | **F31, F32** |
| **Stale docs / never-run tools** | F28, F29, F30 (+ mypy→F26, accurate profile→F27, pre-commit→whitespace) |
| Inspecting shipped artifacts | F21, F22, F23, F24, F25 |
| Running the real pipeline | F13, F14, F15, F16 |
| Writing tests | F18, F19, F20 |
| Live COLMAP docs | F2 |
| Empirical model run | F6 |
| Arithmetic / `doctor` | F5, F8, F12 |
| Code reading alone | F1, F3, F4 |

## State

| Check | Result |
|---|---|
| Tests | **286 passed** (23 files) |
| Lint / format | clean · 85 files |
| Worktree | clean after `ee65708` |
| Double-audit | PASS ×2 at 285 → **re-run needed at 286** |
| Mutation-verified | F4, F19, F20 |

## PS deliverables — actual status

| # | Deliverable | Status |
|---|---|---|
| 1 | Sparse COLMAP model + `sparse.ply` | **produced** |
| 2 | Dense `fused.ply` / georeferenced PLY | **produced** |
| 3 | `mesh.obj` + textures | **produced** (F31) |
| 4 | ENU PLY + `camera_track.geojson`, EPSG-tagged | produced; RMSE needs real GPS |
| 5 | bounds/coverage/scale in `metrics.json` | **produced** (F32 — scale now always reported) |
| 6 | `report.html` + `manifest.json` | **produced** |

## The one remaining item is not code

F7: the bundled sample video has **no GPS** (verified by ffprobe). To close:

```bash
export PATH="$PWD/.tools/colmap-env/bin:$PATH"
uv run drone3d run --config configs/fast.yaml --run-dir outputs/sample_fast --stages georef,metrics,report
python -c "import json;print(json.load(open('outputs/sample_fast/georef/result.json'))['gps_accuracy'])"
```

Compare `rmse_horizontal_m` against **3.0 m**. `tools/synthesize_telemetry.py` is a
fixture generator only — its 0.00 m RMSE is a tautology, not evidence.

## Note for the harness

The question that found F31 and F32 was **"does the pipeline emit what the PS
says it delivers?"** — a different axis from "does the code run" or "do the
artifacts exist". Worth re-asking for each of the 6 deliverables if re-invoked.
Note also that `scale_check` was invisible to tests because the gate was
config-driven: check the **default-path artifact**, not just the configured path.

**Do not** re-run an already-green suite. **Do not write `INFINITY_DONE`** until
criteria 1/2/3/6 are scored against real reference data.
