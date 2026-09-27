# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 19 |
| Phase | AUDIT (cycle 19) |
| What I did | Challenged "none actionable" by running a **type checker for the first time** (`make typecheck` / mypy). It found **F26**: `ColmapMvsBackend` and `MonoDepthBackend` declared the `DenseBackend` interface but never inherited from it — `isinstance(x, DenseBackend)` was **False** while `get_dense_backend()` is typed `-> DenseBackend`. Both fixed; contract test added. |
| What worked | **283 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | 3 missing-import slips in my own test file (fixed immediately); `trimesh` "unresolved" is stub noise, not a defect |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **283** tests (supersedes 281) |

## F26 — the declared return type was untrue

`get_dense_backend()` returns `DenseBackend`, but the two concrete backends
never inherited it. MRO was `[ColmapMvsBackend, object]`. `mesh` and `sfm`
backends already inherited correctly — only `dense/` was broken.

Found by **running mypy** (`make typecheck`), which CI never invokes. It
reported 33 diagnostics in 10 files; **2 were real** (`dense/base.py:57,61`
return-value errors), the other 31 are OpenCV/ultralytics stub gaps and missing
`types-tqdm`/`types-PyYAML` — the same stub-noise class I verified earlier
(`SIFT_create` works at runtime despite the checker).

**Lesson:** the Makefile had a `typecheck` target all along. An unexercised
verification tool is an unverified codebase.

## Defect ledger: 21 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–**F26**, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Running a type checker** | **F26** |
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
last three cycles each found a defect only by using a tool that had never been
exercised:

1. Cycle 17–18: inspecting shipped artifacts (F21–F25)
2. Cycle 19: **running a type checker** (F26)

Remaining unexercised surfaces, if any: `pre-commit` hooks, `make lock`, the
`api`/`geo` extras' runtime paths, and the `accurate` config profile end-to-end.
**Do not** re-run an already-green suite — that is logged as wasted motion.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
