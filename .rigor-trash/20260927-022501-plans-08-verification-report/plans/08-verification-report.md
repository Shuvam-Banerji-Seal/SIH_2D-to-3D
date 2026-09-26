# Final Verification Report

`drone3d` — single-pass drone video → georeferenced 3D (SIH PS 26158 / NTRO)
Generated at cycle 8. Every claim below carries its evidence and a confidence tag.

## 1. How to reproduce the green state

```bash
uv sync --all-extras                      # 7.3 GB; CPython 3.12.8
uv run pytest                             # 233 tests
uv run ruff check . && uv run ruff format --check .

export PATH="$PWD/.tools/colmap-env/bin:$PATH"   # COLMAP 4.2.0 CUDA
uv run drone3d run --config configs/fast.yaml --run-dir outputs/sample_fast \
  --set "ingest.video=datasets/Aerial Views of Rural Riches： Drone Shot of Farmland 🌾🚁 [p8eRmxosalI].webm" \
  --set preprocess.dynamic_masking=true
```

Expected: **8/8 stages OK** (georef skips gracefully without GPS), `report.html`
and `manifest.json` written.

## 2. Test suite

| Check | Result | Evidence |
|---|---|---|
| Tests | **233 passed** | `uv run pytest`, repeated |
| Lint | clean | `ruff check .` |
| Format | clean | `ruff format --check .` (79 files) |
| Coverage | **78%** | `--cov=src/drone3d` |
| CI | would pass | `uv lock --check` ok; CI's exact pytest line executed locally |
| Mutation check | F4 test proven non-vacuous | bug reintroduced → red → reverted → green |
| Double-audit | **PASS ×2** | two consecutive clean waves |

## 3. Pipeline result on the bundled clip

| Stage | Result | Evidence |
|---|---|---|
| ingest | 20 frames @ 1 fps from 19.1 s | `outputs/sample_fast/` |
| preprocess | 20/20 selected, 20 masks written | `preprocess/masks/*.jpg.png` |
| sfm | **21 images, 1286 points**, 0.30 px reprojection | `sfm/result.json` |
| dense | **246,354 points** | `dense/` |
| mesh | **31,883 vertices / 63,904 faces** | `mesh/` |
| georef | 19 tie points, horizontal RMSE **0.00 m** | synthetic GPS (see §4) |
| metrics | 4 clouds summarised | `metrics/metrics.json` |
| report | written | `report.html` |

## 4. Evaluation criteria — evidence matrix

| # | Criterion | Target | Status | Evidence |
|---|---|---|---|---|
| 1 | Geometric accuracy | ≤1×GSD horiz, ≤2×GSD vert | **EVIDENCE OF** | `mean_reprojection_error_px = 0.3019`; 1 GSD ≈ 1 px. Consistent with the target but not proof of ground accuracy. `[VERIFIED: sfm/result.json]` |
| 2 | Metric scale error | ≤2 % | **BLOCKED** | needs real telemetry (F7) |
| 3 | Completeness | ≥80 % @0.5 m | **BLOCKED** | needs reference cloud |
| 4 | Visual quality | facades/roofs recognisable | **PARTIAL** | textured mesh produced; no human review recorded |
| 5 | No dynamic ghosting | visual | **MECHANICALLY VERIFIED** | masks written in COLMAP's convention (name + polarity + location) and passed via `--ImageReader.mask_path`; effect not visually confirmed |
| 6 | Georeferencing | ≤3 m horiz | **BLOCKED** | needs real telemetry. *Measurement* verified via injected-noise test |
| 7 | Robustness | no crash, graceful degradation | **PASS** | 8/8 stages; missing backends skip; Poisson crash falls back to Delaunay |
| 8 | Latency | near-real-time `fast` | **PARTIAL** | ingest 7.2 s, preprocess 2.9 s, sfm 91 s, dense 234 s, mesh 337 s |
| 9 | Usability | one command + report | **PASS** | `drone3d run` → `report.html` |
| 10 | Reproducibility | same input → same metrics | **PASS** | `metrics.json` byte-identical across 4 runs |

**Scored: 3 of 10** (7, 9, 10). **Evidence for 1. Blocked on data: 2, 3, 6.**

## 5. Defects: 15 found and fixed

| ID | Defect | Found by |
|---|---|---|
| F1 | `TelemetrySample.to_dict` slots crash | code read |
| F2 | Masks never fed to COLMAP (location + name + **polarity**) | code read + live COLMAP docs |
| F3 | `_largest_model` precedence | code read |
| F4 | Frame budget ignored when `frame_count<=0` | code read |
| F5 | chamfer ~2.4 GB/chunk | arithmetic |
| F6 | Depth polarity documented backwards | empirical model run |
| F8 | COLMAP absent | `doctor` |
| F12 | `.tools/` broke ruff + 4.4 G untracked | own regression |
| F13 | `model_converter` TXT aborted: output dir missing | running the pipeline |
| F14 | `poisson_mesher` SIGSEGV | running the pipeline |
| F15 | `delaunay_mesher --output_type` rejected | running the pipeline |
| F16 | `write_ply` colour-block **data corruption** | running the pipeline |
| F18 | CSV no-header guard dead code | writing tests |
| F19 | Eager detector map broke *all* feature methods on OpenCV 5.x | writing tests |
| F7 | **Sample video has no GPS** | ffprobe — *data gap, not code* |

**6 of 15 were only findable by executing the real pipeline or writing tests.**
Static code reading found just 2.

## 6. Honest limits

- `tools/synthesize_telemetry.py` derives GPS from the SfM camera centres. Its
  0.00 m RMSE is a **tautology**, not accuracy evidence. It exists so the
  georef path can be exercised and tested.
- Criterion 1's evidence is *reprojection* error — self-consistency with the
  images, not agreement with ground truth. Necessary, not sufficient.
- No human visual review of the textured mesh is recorded (criterion 4).
- `mono_depth` (torch) and the Open3D Poisson path are covered only at their
  guard paths.

## 7. To close the loop

Supply **either** a real drone flight log (CSV / DJI SRT / GPX / JSON) **or** an
NTRO reference cloud, then:

```bash
export PATH="$PWD/.tools/colmap-env/bin:$PATH"
uv run drone3d run --config configs/fast.yaml --run-dir outputs/sample_fast --stages georef,metrics,report
python -c "import json;print(json.load(open('outputs/sample_fast/georef/result.json'))['gps_accuracy'])"
```

Compare `rmse_horizontal_m` against **3.0 m** (criterion 6), then re-run the
double-audit and write `plans/INFINITY_DONE`.
