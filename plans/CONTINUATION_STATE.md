# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 14 |
| Phase | AUDIT (cycle 14) |
| What I did | Found and fixed **F21** by inspecting the *real* `report.html` rather than my synthetic test objects: partial stage re-runs silently stripped earlier stages' metrics from the report. Verified the fix against the real run (all 6 missing numbers restored) and added a regression test. |
| What worked | **270 tests**, ruff clean, 84 files formatted, worktree clean |
| What failed | My appends again stacked duplicate test blocks (caught and deduplicated before commit) |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at 270 tests |

## F21 — found by testing the artifact, not the object

`_collect_metrics()` read only `self._result.stages` — the stages that ran in
*the current invocation*. So every `--stages metrics` re-run (I did several for
reproducibility checks) **rewrote `report.html` without the sfm/dense/mesh
numbers**, even though their `result.json` files were intact on disk.

The real report was missing:
`0.3019` (reprojection), `246354` (dense points), `31883`/`63904` (mesh).

Fix: merge every stage's persisted `result.json`, then overlay the current run's
in-memory metrics. Verified on the real run dir — all six numbers restored.

**Lesson:** my tests asserted on synthetic `PipelineResult` objects and all
passed; only opening the *generated file* exposed this. Artifact-level
verification is not the same as object-level.

## Defect ledger: 21 fixed

F1–F8, F12–F16, F18–F21. **8 of 21 findable only by executing code or inspecting
output artifacts.** F7 is a data gap.

## State

| Check | Result |
|---|---|
| Tests | **270 passed** (20 files) |
| Lint / format | clean · 84 files |
| Coverage | 81% |
| Worktree | clean |
| Double-audit | PASS ×2 at 270 |
| Mutation-verified defects | F4, F19, F20 |
| Reproducibility (criterion 10) | re-verified post-change |

## PS criteria

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criterion 1 has
supporting evidence (0.30 px ≈ 0.3× GSD). Criteria **2, 3, 6 blocked on external data**.

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

Ranked by actual yield over this session:
1. **Running the real pipeline** — found F13–F16.
2. **Inspecting real output artifacts** — found F21 (this cycle).
3. **Writing tests for uncovered modules** — found F18–F20.
4. Mutation-checking those tests — proved they can fail.
5. Sweeping for assertion quality — found 1 tautology.

If re-invoked with no new data, state that the codebase is complete and blocked on
external input. **Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored
against real reference data.
