# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 20 |
| Phase | AUDIT (cycle 20) |
| What I did | Ran the last unexercised surfaces my own state file flagged. `pre-commit run --all-files` (never triggered) fixed trailing whitespace in `.gitignore`. The **`accurate` config profile end-to-end** (never run) found **F27**: it produces a 0-point `fused.ply`, `dense` reports `ok ... 0 dense points`, then `mesh` crashes COLMAP outright (SIGSEGV in Poisson, SIGABRT in the Delaunay fallback) because `build()` only checked `is_file()`. |
| What worked | **284 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | 3 test fixtures used `b"ply"`/`b"mesh"` placeholder bytes and a vertex-0 PLY; the new guard reads the cloud so they had to become real PLYs |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **284** tests (supersedes 283) |

## F27 — an empty dense cloud crashed COLMAP itself

Chain observed on the `accurate` profile:
1. `dense` produced `fused.ply` with **0 points** (229 bytes = header only)
2. `dense` reported **`ok ... 0 dense points`** — success while producing nothing
3. `mesh` passed the `is_file()` check and invoked COLMAP
4. `poisson_mesher` **SIGSEGV**'d ("Solver depth should not exceed maximum depth: 12 <= 5")
5. the Delaunay fallback **SIGABRT**'d (`Percentile<float>` on an empty vector)

Fix: `ColmapMesher.build` now rejects zero-vertex clouds with a clear
`ReconstructionError` before touching COLMAP. Verified against the exact
artifact that crashed it.

**Note:** `dense` still reports `ok` with 0 points — that is arguably correct
(it faithfully ran and fused nothing) and the mesh stage now fails loudly
instead of crashing, so the failure is attributable. Logged, not changed.

## Defect ledger: 22 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F27**, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Exercising an untested surface** | **F26** (mypy), **F27** (accurate profile), `.gitignore` (pre-commit) |
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
| Tests | **284 passed** (23 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 284 |
| Mutation-verified | F4, F19, F20 |
| `make typecheck` | exercised (F26) |
| `pre-commit` | exercised (whitespace fix) |
| `accurate` profile | exercised (F27) |

## PS criteria

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criteria
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

Every verification surface I listed is now exercised. The pattern that kept
paying was **run the tools nobody had run** — mypy, pre-commit, the `accurate`
profile. That source appears exhausted.

Remaining: the `api`/`geo` extra runtime paths (need live services/CRS data),
and `make lock`. Low expected yield. **Do not** re-run an already-green suite.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
