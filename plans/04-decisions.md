# 04 — Decisions

Choices that shape the code, the evidence for each, and the ones that were reversed. Measurements
behind the design are tabulated in [`docs/architecture.md`](../docs/architecture.md) and reproduced
by the scripts in `experiments/`. Last revised 2026-11-22.

## Decisions in force

| ID | Decision | Evidence | Alternative rejected |
|---|---|---|---|
| D1 | **Two profiles.** The fast profile (`configs/fast.yaml`) is the deliverable; the accurate profile (3D Gaussian Splatting, `configs/default.yaml`) is kept for quality studies | The 3DGS profile takes hours for a 55 s clip and yields splats the statement does not ask for, while speed is 20 % of the score ([`docs/roadmap.md`](../docs/roadmap.md)) | One profile that trains splats for every run |
| D2 | **Poses from RAFT flow tracks and pycolmap's global mapper**, not SIFT, in the fast profile | Jal Mahal pass 3: camera centres within 0.2 % of the flight extent and rotations within 0.28° of SIFT SfM, 97/97 registered, 16 s of mapping (`experiments/flow_sfm.py`) | SIFT matching: slower, and the flow is already computed for the keyframes |
| D3 | **Keyframes by measured co-visibility**: the sharpest frame whose overlap with the last keyframe lies in a band | Jal Mahal: 1 fps sampling registered 40 of 55 frames in 2 disconnected models; overlap-band keyframes registered 213 of 213 in one model per group of passes | Fixed-rate sampling |
| D4 | **Direct flow for the "is it 3D?" test**, chained flow only for overlap | Chained RAFT error grows about linearly (0.90 px median at 24 frames, 1.12 px at 32); direct flow stays at 0.12–0.13 px (`experiments/flow_drift.py`) | Chained flow throughout |
| D5 | **Ground-levelled 4-DoF georeferencing** (yaw, scale, translation) | On a straight pass the 7-DoF similarity was 147–221 m off 100 m from the track; the levelled fit 0.2–1.8 m (`experiments/georef_study.py`) | 7-DoF Umeyama on camera centres |
| D6 | **GPU TSDF with a wide truncation band and three confirming views** (two on short shots) | Band 4 → 12 voxels: completeness 0.47 → 0.80; three views instead of Open3D's four: +3–12 points on large models (`experiments/dense_sweep.py`, `tsdf_weight.py`) | Open3D defaults |
| D7 | **One splat trainer at a time** | Jal Mahal's 5 models: 178 s in sequence, 249 s two at once, 264 s three (`experiments/splat_parallel.py`); Vulkan contexts time-slice the GPU | Parallel trainers |
| D8 | **spirula-studio runs as a separate program** | GPL-3.0 against this repository's GPL-2.0-only forbids linking or bundling; invocation through its CLI is the conservative reading | Vendoring or linking it. Relicensing to GPL-2.0-or-later was not pursued (needs the owner's approval) |
| D9 | **Everything the code imports is declared in `pyproject.toml`**; heavy packages sit behind extras and are imported inside the function that uses them | `tifffile` (the GeoTIFF writer) was installed only as a side effect of `scikit-image`; `tests/test_dependencies.py` now fails on any undeclared import | Relying on transitive installs |
| D10 | **Model weights use `/store/huggingface` only if that disk exists** | Setting `HF_HUB_CACHE` to a missing directory made first-use downloads fail with a permission error elsewhere (`utils/model_cache.py`) | A hard-coded default |
| D11 | **`config.py` is not auto-formatted** | `app/schema.py` takes each option's help text from the comment on the field's own line; `ruff format` wrapped 12 such lines and the console lost their help (`test_console_api`) | Formatting it, or parsing the file with a tokenizer (larger change) |
| D12 | **Reported numbers come from committed records through scripts** (`experiments/make_paper.py`, `readme_table.py`), and a record of a run that failed is never reported | The 2-slot throughput record has 13 of 15 jobs failed; both generators would have printed its 6.98× | Hand-copied figures |
| D13 | **The console is a local tool**: it binds 127.0.0.1 by default, has no authentication, and any config key, including `ingest.video`, can be set by a client | `cli.py` (`ui --host`), `app/server.py` (`_deep_set`) | Per-path containment checks, which the override mechanism would bypass; do not expose the port to an untrusted network |

## Reversals

| ID | Was | Became | Why |
|---|---|---|---|
| X1 | 2026-09-26: do not add spirula-studio (not a Python package, GPL-3.0, splats not requested) | 2026-09-27: the accurate profile's SfM, trainer and mesher, as a separate program | The pipeline was rebuilt around the GPU stages; the licence objection is met by D8 |
| X2 | COLMAP command-line SfM, MVS and Poisson/Delaunay meshing with frame pre-processing | Flow-track SfM and GPU TSDF (fast); spirula-studio (accurate) | The scaffold's command-line backends were dropped with the other broken modules on 2026-09-27 (`f75ae60`) |
| X3 | 7-DoF georeferencing | D5 | Roll about a straight track is undetermined |
| X4 | Numeric acceptance targets (GSD multiples, 2 % scale error, 80 % completeness at 0.5 m) | The official criteria and weights only | The targets were not in the official statement ([`docs/problem-statement.md`](../docs/problem-statement.md)) |

## Lessons that still apply

- A round-trip test compares **values**. Writing PLY colours as a separate block passed every shape
  and dtype assertion while corrupting each vertex after the first (`tests/test_ply.py`).
- Code that reads source text turns formatting into behaviour (D11); run the suite after a
  formatter, not only the linter.
- A diagnostic must run where the thing it diagnoses is missing: `drone3d doctor` crashed on an
  install without PyTorch until it stopped importing a PyTorch module for one path constant
  (`tests/test_cli.py::test_doctor_runs_without_torch`).
- A test passes only if it can fail: the dependency and doctor tests were run against the old
  `pyproject.toml` and the old `cli.py` and went red there.

## Pending

| Item | Where |
|---|---|
| Re-run the 2-slot throughput experiment (`uv run python experiments/engine_throughput.py 2`, engine started with `--slots 2`) | [`08`](08-final-verification-report.md), limitation 5 |
| Georeferencing on a real flight log | [`00`](00-understanding.md), Q1 |
