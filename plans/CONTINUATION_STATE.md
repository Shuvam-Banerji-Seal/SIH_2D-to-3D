# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 18 |
| Phase | AUDIT (cycle 18) |
| What I did | Continued artifact inspection and found **F25**: the texture step called `texture_mesher`, which **COLMAP 4.x does not recognise** (it is `mesh_texturer`), passed `--image_path` instead of `--workspace_path`, and looked for `mesh.obj` when the tool writes a textured PLY + atlas. A textured mesh is **criterion 4**. Verified end-to-end: `textured/mesh.ply` now produced. |
| What worked | **281 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | 2 cycles lost to a PATH-less re-run and a missing import in my own tests — both test-side, fixed immediately |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **281** tests (supersedes 279) |

## F25 — the texturing command did not exist

Three faults in one block, all invisible because `configs/fast.yaml` sets
`mesh.texture=false` so the path never executed:

| Fault | Was | Should be |
|---|---|---|
| command name | `texture_mesher` | `mesh_texturer` |
| workspace flag | `--image_path` | `--workspace_path` (the undistorter workspace: `dense_ply.parent`, holding `images/` + `sparse/`) |
| expected output | `mesh.obj` | textured PLY + `texture.png` atlas |

Found by comparing `mesh/result.json` (`textured: false`, empty `textured/` dir)
against the deliverable list, then `colmap help` confirmed the name was wrong.

Verified on the real mesh: `textured/mesh.ply` produced, `textured: true`.

**Correction on my own record:** I first described this as "texturing silently
produced nothing." Wrong — the config had texturing disabled. The real defect is
the latent command-name/flag/output triple.

## Defect ledger: 20 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F25**, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Inspecting shipped artifacts** | **F21, F22, F23, F24, F25** |
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

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criterion 4's
mechanism now actually works (F25) though visual review is still pending.
Criteria **2, 3, 6 blocked on external data**.

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
1. **Inspecting shipped artifacts** — F21, F22, F23, F24, **F25** (five straight)
2. Running the real pipeline — F13–F16
3. Writing tests for uncovered modules — F18–F20

The method that keeps paying: **open what the tool wrote to disk and compare
against what the code promised.** F25 was a wrong command name sitting in a
branch no config enables — unit tests could not see it; only reading
`mesh/result.json` against the deliverable list did.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
