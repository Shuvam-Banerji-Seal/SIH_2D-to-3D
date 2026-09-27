# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 17 |
| Phase | AUDIT (cycle 17) |
| What I did | Continued the artifact-inspection method and found **F24**: `DynamicMasker` promised "vehicles, people and animals" but used MOG2 background subtraction, which on a moving UAV flags the *whole static scene* as foreground — measured **67–74% of early frames** on the bundled farmland clip. Fixed with a `max_dynamic_fraction` guard. |
| What worked | **279 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | Nothing of substance; 2 missing-import slips in my own tests (fixed immediately) |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **279** tests |

## F24 — the mask was eating the scene, not the movers

Found by inspecting the **emitted masks** (the F21–F24 pattern): frame 0 was 100%
keep, frames 1–5 were ~70% ignore as one 628k-pixel blob. That is not a mover —
it is camera motion.

Root cause: `cv2.createBackgroundSubtractorMOG2` assumes a static camera. On a
translating UAV every pixel changes, so the "foreground" is the terrain.

Fix: `max_dynamic_fraction=0.5`. A mask covering more than half the frame is
camera motion, not movers, so it is dropped for that frame.

Verified on the real clip:

| | un-guarded | guarded |
|---|---|---|
| frames 1–5 (camera motion) | 67–74% dropped | **0%** |
| frames 6–7 (plausible movers) | 13.9% / 15.0% | preserved unchanged |

**This is criterion 5's intent** — mask dynamic objects without discarding the
scene. Prior to F24 the feature did the opposite.

## Defect ledger: 19 code defects fixed + 1 data gap

F1–F6, F8, F12–F16, F18–F24, plus F7 (data gap).

| Found by | Defects |
|---|---|
| **Inspecting emitted/shipped artifacts** | **F21, F22, F23, F24** |
| Running the real pipeline | F13, F14, F15, F16 |
| Writing tests | F18, F19, F20 |
| Live COLMAP docs | F2 |
| Empirical model run | F6 |
| Arithmetic / `doctor` | F5, F8, F12 |
| Code reading alone | F1, F3, F4 |

## State

| Check | Result |
|---|---|
| Tests | **279 passed** (21 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 279 |
| Mutation-verified | F4, F19, F20 |

## PS criteria

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criterion 5
now has a *mechanism* that behaves correctly (verified on real data). Criteria
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
1. **Inspecting emitted/shipped artifacts** — F21, F22, F23, **F24** (four straight)
2. Running the real pipeline — F13–F16
3. Writing tests for uncovered modules — F18–F20

The method that keeps paying: **look at what the tool actually wrote to disk** —
masks, reports, manifests — and compare against what the code promised. F24 was
invisible to unit tests and to `pytest`; only opening the PNG masks revealed that
70% of the scene was being deleted.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
