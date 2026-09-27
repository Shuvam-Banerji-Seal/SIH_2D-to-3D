# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 18 |
| Phase | AUDIT (cycle 18) |
| What I did | Artifact inspection found **F25**: the texture step invoked `texture_mesher`, a command COLMAP 4.x does not recognise (it is `mesh_texturer`), passed `--image_path` where the tool wants `--workspace_path`, and looked for `mesh.obj` when it writes a textured PLY + atlas. Verified end-to-end against the real mesh. |
| What worked | **281 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | My own workspace heuristic was wrong on the first try (used `images_dir.parent`; the tool wants `dense_ply.parent`) — corrected against the real command |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **281** tests |

## F25 — texturing called a command that does not exist

Three faults in one block, none previously exercised because
`configs/fast.yaml` sets `mesh.texture=false`:

| | old | correct |
|---|---|---|
| command | `texture_mesher` | **`mesh_texturer`** |
| workspace flag | `--image_path` | **`--workspace_path`** (the undistorter workspace: `dense_ply.parent`) |
| expected output | `mesh.obj` | **textured PLY + texture atlas PNG** |

A textured mesh is **criterion 4**, so this path matters. Found by comparing
`mesh/result.json` (`"textured": false`, empty `textured/` dir) against the PS
deliverable list, then confirming via `colmap help` that the command name was
simply wrong.

Verified end-to-end on the real mesh: `textured/mesh.ply` now produced,
`textured: true`.

## Defect ledger: 20 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F25**, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Inspecting shipped output artifacts** | **F21, F22, F23, F24, F25** (five consecutive) |
| Running the real pipeline | F13, F14, F15, F16 |
| Writing tests | F18, F19, F20 |
| Live COLMAP docs | F2 |
| Empirical model run | F6 |
| Arithmetic / `doctor` | F5, F8, F12 |
| Code reading alone | F1, F3, F4 |

## State

| Check | Result |
|---|---|
| Tests | **281 passed** (22 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 281 |
| Mutation-verified | F4, F19, F20 |

## PS criteria

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criterion 4
now has a working texturing path (F25) — quality review still pending. Criteria
**2, 3, 6 blocked on external data**.

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

In-repo work is exhausted. **Re-running an already-green suite is not progress.**

Ranked by actual yield across this session:
1. **Inspecting shipped output artifacts** — F21, F22, F23, F24, **F25** (five straight)
2. Running the real pipeline — F13–F16
3. Writing tests for uncovered modules — F18–F20

F25 is the sharpest example: three wrong facts about an external tool (command
name, flag, output filename) survived every unit test because the path was
disabled in the default config. Only *comparing a result file against the
deliverable list* exposed it.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
