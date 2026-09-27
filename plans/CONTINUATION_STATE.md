# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 16 |
| Phase | AUDIT (cycle 16) |
| What I did | Found and fixed **F23** — `manifest.json` recorded only the current invocation's stages, so a `--stages metrics` re-run wrote a manifest claiming 2 stages and 3 artifacts while six `result.json` files sat on disk. Same partial-re-run trap as F21, on a different artifact I had never inspected. |
| What worked | **277 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | Nothing of substance |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **277** tests |

## F23 — the manifest had the same bug as the report

`_write_manifest` serialised `self._result.to_dict()` — only stages that ran in
*this* invocation. Partial re-runs therefore produced a manifest that
under-reported the run directory.

Fix: stages not run here are reconstructed from their persisted `result.json`
plus every file in their stage directory. Verified on the real run dir:
a `--stages metrics` re-run now yields **7 stages / 20 artifacts** instead of
2 stages / 3 artifacts.

**Pattern worth reusing:** F21, F22 and F23 were all the same failure mode —
*partial re-runs silently degrading run-directory artifacts* — and all three
were invisible to unit tests on synthetic objects. They were found by
**opening the shipped artifacts** (`report.html`, `manifest.json`) and comparing
them against what the code promised.

## Defect ledger: 18 code defects fixed + 1 data gap

| ID | Defect | Found by |
|---|---|---|
| F1, F3 | slots crash; precedence | code read |
| F2 | masks never fed to COLMAP (location + name + polarity) | code read + live docs |
| F4 | frame budget ignored when `frame_count<=0` | code read |
| F5 | chamfer ~2.4 GB/chunk | arithmetic |
| F6 | depth polarity documented backwards | empirical model run |
| F8 | COLMAP absent | `doctor` |
| F12 | `.tools/` broke ruff | own regression |
| F13–F16 | TXT mkdir, poisson SIGSEGV, `--output_type`, PLY corruption | **running the pipeline** |
| F18–F20 | CSV guard, eager detector map, double-escaping | **writing tests** |
| F21–F23 | **partial re-runs degraded report/manifest** | **inspecting shipped artifacts** |
| F7 | sample video has no GPS | ffprobe — **data gap** |

## State

| Check | Result |
|---|---|
| Tests | **277 passed** (21 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 277 |
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

In-repo work is exhausted. **Re-running an already-green suite is not progress.**

Ranked by actual yield across this session:
1. **Inspecting shipped output artifacts** — F21, F22, F23 (all three in the last
   three cycles, all invisible to unit tests)
2. Running the real pipeline — F13–F16
3. Writing tests for uncovered modules — F18–F20
4. Mutation-checking those tests

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
