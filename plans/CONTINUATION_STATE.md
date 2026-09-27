# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 20 |
| Phase | AUDIT (cycle 20) |
| What I did | Ran the last unexercised surfaces my own state file flagged. `pre-commit` (never triggered) fixed trailing whitespace in `.gitignore`. The **`accurate` config profile end-to-end** (never run) found **F27**: it produces a 0-point `fused.ply`, `dense` reports `ok ... 0 dense points`, then `mesh` **crashes COLMAP itself** (SIGSEGV in Poisson, SIGABRT in the Delaunay fallback) because `build()` only checked `is_file()`. |
| What worked | **284 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | My scripted PLY-placeholder rewrite left 2 files unformatted (caught by `ruff format --check`, fixed) |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **284** tests (supersedes 283) |

## F27 — empty dense cloud crashed COLMAP outright

Observed on the `accurate` profile (the surface nobody had exercised):

```
dense: OK  0 dense points        <- reports success while producing nothing
mesh:  FAILED  poisson_mesher exit -11 (SIGSEGV, "Solver depth ... 12 <= 5")
              delaunay fallback exit -6 (SIGABRT, Percentile on empty vector)
```

Root cause: `ColmapMesher.build` checked `dense_ply.is_file()` but an **empty**
cloud passes that. `fused.ply` was 229 bytes of header, 0 vertices.

Fix: reject zero-vertex clouds up front with a clear `ReconstructionError`.
Test fixtures that stubbed `fused.ply` with raw `b"ply"` bytes had to become
real PLYs too, since the guard reads the vertex count.

Note: `dense` still reports `ok` with 0 points. That is arguably correct — it
fused nothing but did not fail — and `mesh` now fails *legibly* instead of
crashing. The crash, not the empty result, was the defect.

## Defect ledger: 22 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F27**, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Exercising untested surfaces** | **F26** (mypy), **F27** (accurate profile), whitespace (pre-commit) |
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
| Exercised | `make typecheck`, `pre-commit`, `accurate` profile |

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

Every verification surface I could name is now exercised. The yield is
declining: mypy (F26), pre-commit (whitespace), `accurate` profile (F27) found
things — but the last three artifact inspections went clean.

**Do not** re-run an already-green suite. If re-invoked, the only honest options
are (a) exercise a surface nobody has named yet, or (b) state that the codebase
is complete and blocked on external input.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
