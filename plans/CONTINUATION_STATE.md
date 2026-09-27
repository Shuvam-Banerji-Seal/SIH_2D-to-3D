# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 21 |
| Phase | AUDIT (cycle 21) |
| What I did | Checked whether the pipeline actually produces the **PS deliverables**, not just runs. Found **F31**: deliverable 3 names `mesh.obj`, but no code path ever emitted it — `ColmapMesher` wrote only `mesh.ply`+`texture.png`, and the one caller of `convert_mesh_format` (`Open3DMesher`) wrote `mesh.glb`. The OBJ capability existed and was simply never wired. Now emitted on the default path. Also fixed 3 stale artifact-path docs (F30 class). |
| What worked | **285 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | One `has_trimesh` import slip in my own edit (fixed immediately) |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **285** tests (supersedes 284) |

## F31 — the named deliverable was never produced

`docs/problem-statement.md` deliverable 3: *"Textured 3D mesh — `mesh.obj` + textures
(`mesh.ply`, `mesh.glb`)"*.

| Path | Wrote | `mesh.obj`? |
|---|---|---|
| `ColmapMesher` (default) | `mesh.ply` + `texture.png` | **no** |
| `Open3DMesher` (`mesh` extra) | `mesh-open3d.ply` + `mesh.glb` | **no** |
| `convert_mesh_format` | supports OBJ | never called for it |

Fix: `ColmapMesher` now converts the finished PLY → `mesh.obj` when trimesh is
available, recorded as `metadata.obj_path`. Verified end-to-end on the real
mesh: **2.78 MB `mesh.obj`** with `mtllib`/`usemtl`/`vt` (61,274 verts, 73,823 faces).

**Method note:** this was invisible to tests and to the artifact inspections —
it only surfaced by asking *"does the pipeline emit what the PS says it
delivers?"* rather than *"does the code run?"*

## Defect ledger: 24 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F31** (F9–F11 never existed), plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Checking deliverables against the PS** | **F31** |
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
| Tests | **285 passed** (23 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 285 |
| Mutation-verified | F4, F19, F20 |

## PS deliverables — actual status

| # | Deliverable | Status |
|---|---|---|
| 1 | Sparse COLMAP model + `sparse.ply` | **produced** |
| 2 | Dense `fused.ply` / georeferenced PLY | **produced** |
| 3 | `mesh.obj` + textures | **produced** (F31) |
| 4 | ENU PLY + `camera_track.geojson`, EPSG-tagged | produced; RMSE needs real GPS |
| 5 | bounds/coverage/scale in `metrics.json` | **produced** |
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

The question that found F31 was **"does the pipeline emit what the PS says it
delivers?"** — a different axis from "does the code run" or "do the artifacts
exist". Worth re-asking for each of the 6 deliverables if re-invoked.

**Do not** re-run an already-green suite. **Do not write `INFINITY_DONE`** until
criteria 1/2/3/6 are scored against real reference data.
