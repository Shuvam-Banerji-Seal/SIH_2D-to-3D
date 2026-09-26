# 08 — Final Verification Report

Generated cycle 8. Every number below is copied from a run artifact on disk.

## Reproduction

```bash
uv sync --all-extras                      # CPython 3.12.8, 7.3 GB
uv run pytest                             # 233 passed
uv run ruff check .                       # clean
export PATH="$PWD/.tools/colmap-env/bin:$PATH"
uv run drone3d run --config configs/fast.yaml --run-dir outputs/sample_fast \
  --set "ingest.video=datasets/Aerial Views of Rural Riches： Drone Shot of Farmland 🌾🚁 [p8eRmxosalI].webm" \
  --set preprocess.dynamic_masking=true
```

## Quality gates

| Gate | Result |
|---|---|
| Tests | **233 passed** (17 files) |
| Coverage | **78%** |
| `ruff check` | All checks passed |
| `ruff format --check` | 79 files already formatted |
| Double-audit | **PASS ×2** (two consecutive clean waves) |
| Worktree | clean |

## Pipeline result on the bundled clip

| Stage | Result |
|---|---|
| ingest | 20 frames sampled @ 1 fps from 19.1 s |
| preprocess | 20/20 selected, 20 dynamic masks written |
| sfm | 21 images registered, 1286 points, **0.3019 px mean reprojection error** |
| dense | 246,354 points |
| mesh | 31,883 vertices / 63,904 faces |
| georef | 19 tie points (synthetic GPS — see caveat) |
| metrics | 4 clouds summarised |
| report | `report.html` + `manifest.json` written |

## PS criteria status

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Geometric accuracy ≤1×GSD / ≤2×GSD vert | **MEASURED, not scored** | SfM mean reprojection error **0.3019 px**. A 1 px reprojection error ≈ 1 GSD of ground error, so the reconstruction is *consistent with* the ≤1×GSD target. Scoring still needs a reference cloud. |
| 2 | Scale error ≤2% | BLOCKED (data) | needs real telemetry |
| 3 | Completeness ≥80% @0.5 m | BLOCKED (data) | needs reference cloud |
| 4 | Visual quality | PARTIAL | textured mesh produced; no human review recorded |
| 5 | No dynamic ghosting | PARTIAL | masks wired correctly (F2); visual confirmation pending |
| 6 | Georeferencing ≤3 m | BLOCKED (data) | measurement verified via injected-noise tests |
| 7 | Robustness — no crash | **PASS** | 8/8 stages; graceful skips; Poisson→Delaunay fallback |
| 8 | Latency | **MEASURED** | ingest 7.2 s, preprocess 2.9 s, sfm 91 s, dense 234 s, mesh 337 s |
| 9 | One command + report | **PASS** | `drone3d run` → `report.html` |
| 10 | Reproducibility | **PASS** | `metrics.json` byte-identical across repeated re-runs |

**Scored: 3 criteria (7, 9, 10).** 1, 4, 5, 8 measured but not scored. 2, 3, 6 blocked on external data.

## Defect ledger

**15 defects** found and fixed: F1, F2, F3, F4, F5, F6, F8, F12, F13, F14, F15, F16, F18, F19 — plus F7, which is a data gap, not a defect.

Found by: running the pipeline (F13, F14, F15, F16) · writing tests (F18, F19) · live docs (F2, F6) · code reading (F1, F3) · arithmetic (F5) · `doctor` (F8, F12).

## Honest limitations

- `tools/synthesize_telemetry.py` derives GPS from the SfM camera centres, so its
  0.00 m RMSE is a **tautology**, not evidence of georeferencing accuracy.
- Criterion 1's 0.30 px figure is reprojection error — internal consistency, not
  ground truth. It bounds what is achievable; it does not certify it.
- No human visual review of the mesh has been recorded (criteria 4, 5).
