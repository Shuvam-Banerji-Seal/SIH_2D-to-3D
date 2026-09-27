# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 19 |
| Phase | AUDIT (cycle 19) |
| What I did | Challenged "nothing actionable" by running **mypy for the first time** (the `Makefile` has a `typecheck` target CI never invokes). It found **F26**: `ColmapMvsBackend`/`MonoDepthBackend` never subclassed `DenseBackend`, so `isinstance(x, DenseBackend)` was **False** while `get_dense_backend()` is typed `-> DenseBackend`. Fixed both + added a contract test. |
| What worked | **283 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | My own test needed 3 missing imports (fixed immediately) |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **283** tests |

## F26 — the declared return type was a lie

`get_dense_backend()` returns `DenseBackend`, but both dense backends had
`MRO = [X, object]` — they duck-typed the interface without inheriting it.
`isinstance` checks therefore failed. The `mesh` and `sfm` backends were correct.

Found by running **mypy**, which CI never invokes (`typecheck` is not in
`.github/workflows/ci.yml`). 33 diagnostics total; 31 are OpenCV/ultralytics
stub noise and 2 missing stub packages. Only these 2 were real.

**Lesson:** an unexercised verification tool is an unverified codebase. The
Makefile advertised `typecheck` and nobody had ever run it.

## Defect ledger: 21 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F26**, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Running a never-used tool (mypy)** | **F26** |
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
| Tests | **283 passed** (23 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 283 |
| Mutation-verified | F4, F19, F20 |
| `make typecheck` | now exercised (was never run before) |

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

In-repo work is exhausted *unless* a new verification **surface** is found. The
last three cycles each found a defect by using a tool nobody had exercised:
F21–F25 via artifact inspection, F26 via mypy.

Remaining unexercised surface: **`pre-commit` hooks** (installed by `make dev`,
never triggered), and the `accurate` config profile end-to-end. If re-invoked,
try those before concluding. **Do not** re-run an already-green suite.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
