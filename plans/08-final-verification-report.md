# 08 — Verification report

`drone3d`: single-pass drone video → georeferenced 3D model (SIH problem statement 26158, NTRO).
Every figure below is either run in the revision of 2026-11-22 or computed from a committed record
named beside it. The machine used for the test runs has **no NVIDIA GPU**; GPU behaviour is
therefore reported from the A100 benchmark record, not re-measured here.

## 1. Reproduce

```bash
uv sync --all-extras                        # core, dev and every optional backend (PyTorch 2.14, CUDA 13.2)
uv run pytest                               # tests that need a GPU skip themselves without one
uv run ruff check src tests && uv run ruff format --check src tests
uv run drone3d doctor                       # what is installed, what is missing

tools/fetch_datasets.sh                     # the fifteen sample videos listed in resources.md
uv run drone3d run --config configs/fast.yaml --set ingest.video=datasets/<video>.webm
uv run python experiments/collect_system.py && uv run python experiments/readme_table.py --write
```

## 2. Quality gates

| Check | Result |
|---|---|
| Tests, all extras installed (PyTorch 2.14+cu132, no CUDA device) | **289 passed, 20 skipped**; every skip needs CUDA |
| Tests, core and dev dependencies only (CI's environment) | **201 passed, 29 skipped**; skips are the tests needing PyTorch, Open3D, trimesh, laspy |
| Lint and format | `ruff check` and `ruff format --check` clean for `src` and `tests`; `ruff check` clean for `experiments`, `tools`, `promo` |
| Lockfile | `uv sync --locked` succeeds; no surviving package changed version |
| CI smoke steps | package imports; `drone3d doctor` exits 0 on a core-only install; all four `configs/*.yaml` load |
| Can the new tests fail? | The `doctor` test is red on the previous `cli.py` (it imported PyTorch for a path); the dependency test is red on the previous `pyproject.toml` (names `tifffile`) |

## 3. Measured results

Fifteen sample videos, fast profile, warm engine, one video at a time, one A100 80 GB, other
processes holding at most 2.8 GB of the card (`paper/figures/all_maps.json`; per-video table in the
README).

| Quantity | Value |
|---|---|
| Footage | 29.3 min over 15 videos |
| Within the budget (1.5 × video length, i.e. 15 min per 10 min) | **11 of 15**; the misses are three FPV flights and a 19 s clip, whose per-run costs dominate a budget proportional to length |
| Processing time per video second | 0.64 on the 11 within budget; 0.90 over all 15 |
| View completeness (share of the registered keyframes' non-sky pixels the mesh covers) | median **0.88**, range 0.48–0.95; lowest where the far field lies beyond three times the triangulated range (PNB Merdeka 0.48, Cristo Redentor 0.49, Qutub Minar 0.51) |
| Registration | every keyframe registered except Qutub Minar (106 of 113) |

## 4. Georeferencing with synthetic GPS

The sample videos carry no flight log, so GPS is simulated from the true track with 1.5 m horizontal
and 3.0 m vertical noise on the six Jal Mahal models (`experiments/georef_e2e.py`,
`paper/figures/fast_georef_e2e.json`). Leave-one-out horizontal RMSE is 1.9–2.7 m against the 1.5 m noise.

| Model | Rotation error (°) | Scale error (%) | Geometry within 100 m of the track | Median mesh error within 100 m (m) | Median mesh error beyond 300 m (m) |
|---|---|---|---|---|---|
| 0 | 0.015 | 0.35 | 0.1 % | 2.06 | 7.3 |
| 1 | 0.226 | 0.07 | 0.1 % | 0.86 | 8.7 |
| 2 | 0.360 | 0.44 | 0 % | — | 68.8 |
| 3 | 0.019 | 0.47 | **93.6 %** | **0.43** | — |
| 4 | 0.337 | 0.50 | 0 % | — | 17.8 |
| 5 | 0.056 | 0.50 | no mesh | — | — |

Rotation and scale are recovered to 0.36° and 0.5 % on every model. The mesh error is a property of
*where the geometry lies*: it is about half a metre where the model is near the track and grows with
distance. Only model 3 has most of its geometry (94 %) within 100 m; the others lie mostly 300 m or
more from the track, with medians of 7–69 m. The official target of 1 m is thus met only near the
track, and the figure "below 1 m within 100 m" describes one model, not the set.

## 5. Evidence against the official criteria

| Criterion (weight) | Evidence | Gap |
|---|---|---|
| Reconstruction accuracy (30 %) | Poses from flow within 0.2 % of the flight extent and 0.28° of SIFT SfM (Jal Mahal pass 3); georeferencing as in §4; held-out depth error 2.7 % after calibration (accurate profile) | No real flight log and no reference cloud: ≤ 1 m is unverified away from the track |
| Model completeness (20 %) | Median view completeness 0.88 over 15 videos (§3) | Measured against what the keyframes saw, not a reference model; unseen surfaces are absent |
| Processing speed (20 %) | 11 of 15 within budget; 0.64 s per video second on those (§3) | FPV flights and very short clips exceed a budget proportional to length |
| Innovation (15 %) | Overlap-band keyframes, parallax verdict, structure from flow, ground-levelled 4-DoF georeferencing, labelled generated and complete models ([`04`](04-decisions.md), [`paper/`](../paper/)) | Qualitative |
| Scalability (10 %) | Warm engine with job slots; live segments modelled 26–38 s after a 30 s segment closes (Qutub Minar replay) | Multi-slot throughput is unmeasured, limitation 5 |
| User interface (5 %) | Web console and explorer; `tools/record_ui/e2e.mjs` drives every page and layer in a GPU browser | Not run in this revision |

Output formats required by the statement: OBJ, PLY, LAS, GeoTIFF, GLB/glTF, FBX. Round trips of LAS,
GeoTIFF (scale, tie point, CRS), GLB and the mesh writers are tested (`test_export_formats`); FBX
is written by the optional `meshconv` converter (`tools/build_meshconv.sh`) and is not covered by the
test suite.

## 6. Defects found and fixed in this revision

| Defect | Effect | Guard |
|---|---|---|
| `drone3d doctor` imported a PyTorch module to read one path | Traceback on an install without PyTorch, including CI's smoke step | `test_doctor_runs_without_torch` |
| `tifffile` imported by the GeoTIFF writer but declared nowhere | GeoTIFF export failed without the `depth` extra; `scipy` and `pillow` were likewise only in `depth` | `test_dependencies` |
| Engine capability probe ignored `$DEPTH_ASSETS_DIR` | Marigold reported present or absent contrary to what the loader would do | `test_model_paths` |
| `/store/huggingface` forced as the weight cache | First-use downloads failed with a permission error on any other machine | `test_model_paths` |
| `drone3d view` edited the standard library's shared MIME table | Leaked into every HTTP server in the process | `test_cli` |
| Tests needing PyTorch, Open3D, trimesh or laspy failed instead of skipping | pytest aborted in CI's environment | 201 passed on core and dev only |
| README "Measured" section empty in every commit | The benchmark the docs refer to was missing | filled from `all_maps.json` |
| Throughput generators reported a run whose jobs failed | 6.98× "throughput" from 13 failed jobs of 15 | both generators skip failed runs |
| `ruff format` wrapped the comments `app/schema.py` reads | 12 options lost their help text in the console | `test_console_api`; `config.py` exempted from the formatter |

## 7. Limitations

1. **No real flight log or reference cloud.** Accuracy is verified with synthetic GPS only (§4).
2. Completeness is a view measure, not a comparison with ground truth.
3. Dynamic objects are not masked.
4. GPU code (RAFT with CUDA Graphs, TSDF fusion, texture baking, nvJPEG) was not executed in this
   revision; 20 tests skip. It rests on the A100 benchmark record.
5. `paper/figures/engine_throughput.json` records a 2-slot run in which 13 of 15 jobs failed. No
   throughput claim is made from it until it is repeated.
6. The accurate profile needs the spirula-studio build and the Marigold weights, neither present on
   the verification machine (`drone3d doctor` shows them as missing); it was not re-run.
7. The wording "below 1 m within 100 m of the track" in the README, `docs/problem-statement.md` and
   the paper should carry the model-3 qualification of §4.

## 8. To close the loop

With a real flight log (CSV, DJI SRT, GPX or JSON). The log is loaded by `ingest` and joined to the
keyframes by timestamp, so run the profile with it rather than re-running `georef` alone:

```bash
uv run drone3d run --config configs/fast.yaml --set ingest.video=<video> --set ingest.telemetry=<log>
```

then compare `models[*].in_sample.rmse_horizontal_m` and `models[*].held_out.loo_rmse_horizontal_m`
in `outputs/<run>/georef/result.json` with the official 1 m. To repeat the throughput run, start the
engine with `drone3d engine --slots 2`, run `uv run python experiments/engine_throughput.py 2`,
check that its `failed` list is empty, and regenerate the README table with `readme_table.py --write`.
