# CONTINUATION STATE

## Session Summary

| Field | Value |
|-------|-------|
| Session # | 17 |
| Phase | AUDIT (cycle 17) |
| What I did | Inspected the emitted dynamic masks — the artifact-inspection method that has now found 4 defects — and found **F24**: `DynamicMasker` promises "vehicles, people and animals" but uses MOG2 background subtraction, which on a translating UAV flags the whole static scene. Measured 67–74% of early frames ignored. Added a `max_dynamic_fraction` guard (verified: 67–74% → 0%, plausible 13–23% detections preserved). |
| What worked | **279 tests**, ruff clean, 85 files formatted, worktree clean |
| What failed | Nothing of substance |
| Errors remaining | **F7 only — external data gap** |
| Next priorities | **None actionable in code.** |
| Blockers | Real flight log / NTRO reference cloud |
| Audit status | **DOUBLE_PASS** — waves A & B at **279** tests |

## F24 — the masker ate the scene it was meant to protect

`DynamicMasker` uses `cv2.createBackgroundSubtractorMOG2`, which assumes a
**static camera**. On a translating UAV every pixel changes, so static terrain is
flagged as foreground. Measured on the bundled farmland clip: frames 1–5 dropped
**67–74%** of their pixels (a single 628k-pixel blob — the whole scene, not a
mover), frames 6+ dropped 13–23%. That silently deletes valid SfM features — the
opposite of criterion 5's intent.

Fix: `max_dynamic_fraction=0.5`. A mask covering more of the frame than that is
camera motion, not movers, so it is dropped for that frame and the scene kept.

Verified on the real clip: warm-up frames **0%** dropped (was 67–74%), plausible
detections preserved byte-identical.

**Method note:** found by *reading the emitted mask PNGs* — checking each one's
value histogram and blob sizes. This is the 4th consecutive defect invisible to
unit tests on synthetic objects.

## Defect ledger: 19 code defects fixed + 1 data gap

| ID | Defect | Found by |
|---|---|---|
| F1, F3, F4 | slots crash; precedence; frame budget | code read |
| F2 | masks never fed to COLMAP (location + name + polarity) | code read + live docs |
| F5, F8, F12 | chamfer memory; COLMAP absent; `.tools/` broke ruff | arithmetic / `doctor` |
| F6 | depth polarity documented backwards | empirical model run |
| F13–F16 | TXT mkdir; poisson SIGSEGV; `--output_type`; PLY corruption | **running the pipeline** |
| F18–F20 | CSV guard; eager detector map; double-escaping | **writing tests** |
| F21–F23 | partial re-runs degraded report/manifest | **inspecting artifacts** |
| **F24** | **dynamic masking deleted 67–74% of the scene** | **inspecting emitted masks** |
| F7 | sample video has no GPS | ffprobe — **data gap** |

## State

| Check | Result |
|---|---|
| Tests | **279 passed** (21 files) |
| Lint / format | clean · 85 files |
| Worktree | clean |
| Double-audit | PASS ×2 at 279 |
| Mutation-verified | F4, F19, F20 |

## PS criteria

Scored **3 of 10** (7 robustness, 9 usability, 10 reproducibility). Criteria
**2, 3, 6 blocked on external data**. Criterion 5 now has a *correct* mask
mechanism (F2 wiring + F24 guard), though visual confirmation still pending.

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
1. **Inspecting shipped output artifacts** — F21, F22, F23, **F24** (four consecutive)
2. Running the real pipeline — F13–F16
3. Writing tests for uncovered modules — F18–F20
4. Mutation-checking those tests

The method that keeps paying: **open the files the tool actually produced and
compare against what the code promised.** Unit tests on synthetic objects missed
all four of F21–F24.

**Do not write `INFINITY_DONE`** until criteria 1/2/3/6 are scored against real
reference data.
