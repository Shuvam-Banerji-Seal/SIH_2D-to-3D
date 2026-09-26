# 04 — Decisions / Discoveries / Reversals / Open Qs

Last updated: 2026-09-26 (cycle 8).

## Decisions

| ID | Decision | Rationale | Rejected |
| --- | --- | --- | --- |
| D1 | **Do not add spirula-studio now** | Not a `uv`/Python dep (S1); GPL-3.0 vs our GPL-2.0-only forbids bundling (S2); PS never asks for splatting (S3); duplicates 4/8 stages (S4); would starve `georef` of its COLMAP inputs (S5) | "Add it as our SfM backend" — rejected, see D1 rationale S5 |
| D2 | **Fix F8 via COLMAP** (apt 3.7-2 / conda-forge 4.2.0) | Cheapest path: outputs exactly the layout `sfm`+`georef` already parse; BSD-3-Clause ⇒ no licence issue; zero adapter code | Installing Spirula instead — rejected (would regress criteria 2 & 6) |
| D3 | Spirula **parked** as a future optional `splat` stage / mesh backend; if pursued → **subprocess only, never bundled** | Preserves our GPL-2.0-only licence; A100 present and unused; re-open on any of: COLMAP unavailable to judges, deliberate 3DGS bonus demo, or confirmed COLMAP-*write* | Vendoring the source / linking — rejected (S2) |
| D4 | Relicence to `GPL-2.0-or-later` **not** pursued this cycle | Owner approval required; arm's-length invocation keeps us safe regardless | Relicence-then-integrate — logged only |
| D5 | `uv init` **not run** on the repo | Project already initialized; `uv` refuses by design. Verified non-destructive in an isolated `/tmp` probe first | Running `uv init` blind — risked `README.md` clobber (A9) |

## Discoveries

| ID | Finding | Impact | Status |
| --- | --- | --- | --- |
| S1 | Spirula is a C++/Vulkan binary, "Nothing is left in Python" | `uv` cannot install it | verified |
| S2 | Licences: theirs **GPL-3.0**, ours **GPL-2.0-only** | Bundling/linking forbidden; subprocess OK | verified |
| S3 | PS + code contain **zero** splatting/NeRF references | 3DGS is not a scored deliverable | verified |
| S4 | Spirula covers ingest/preprocess/masking/SfM/mesh | Overlap ⇒ adapter work, not saved work | verified |
| S5 | Spirula *parses* COLMAP; no evidence it *writes* it | Would break our `georef` stage inputs | **[UNKNOWN]** |
| S6 | Single maintainer (bus factor 1) | Continuity risk | verified |
| S7 | COLMAP: apt `3.7-2`, conda-forge `4.2.0` (cpu/cuda_129/cuda_130) | D2 is executable today | verified |
| S8 | `MeshExport.h` writes **PLY/OBJ/glTF/GLB/STL** | PLY is compatible with our `mesh/` contract | verified |
| E1 | `uv init` errors on an initialized project, files untouched | Request half (b)-init is a no-op | verified (executed) |
| E2 | `uv sync` → `.venv` CPython **3.12.8**, 22 pkgs, `drone3d` editable | Env now usable | verified (executed) |
| E3 | `ruff check` + `ruff format --check` both **green** (54 files) | Repo is lint-clean pre-existing | verified (executed) |
| E4 | `pytest` → *no tests ran* | **Zero tests** still true post-sync | verified (executed) |
| F13 | `model_converter --output_type TXT` aborts unless the output dir **already exists** (PLY output has no such requirement) | `sparse_txt` was never created ⇒ SfM failed at its final step | verified (executed) |
| F14 | `poisson_mesher` **SIGSEGVs** in COLMAP's surface trimmer on this cloud | mesh stage crashed; Delaunay succeeds on the same input | verified (executed) |
| F15 | `delaunay_mesher` rejects `--output_type` outright in COLMAP 4.x | the Delaunay branch **never worked**; format is inferred from the `.ply` extension | verified (executed) |
| F16 | `write_ply` wrote colours as a separate contiguous block; PLY binary is **interleaved per vertex** | every georeferenced cloud round-tripped corrupt past vertex 1 | verified (executed) |
| F18 | CSV no-header guard was dead code — `csv.DictReader` reports `[]`, not `None` | blank CSV silently yielded zero samples | verified (executed) |
| F19 | Feature detector map was built **eagerly**; AKAZE is gone in OpenCV 5.x | *every* `detect_features` call crashed, not just `akaze` | verified (executed) |
| E5 | COLMAP 4.2.0 emits `frames.txt` + `rigs.txt` alongside the legacy model files | our parsers tolerate the extra files | verified (executed) |
| E6 | SfM reports **0.30 px mean reprojection error** on the sample clip | strong geometric consistency; the number is trustworthy once GPS arrives | verified (executed) |
| E7 | CI's pytest step is **not** a no-op now — its exact invocation passes (233 tests, 78% cov) | CI is green-end-to-end verified | verified (executed) |
| E5 | GPU **A100 80GB**, driver **610.43.02**, `nvidia_icd.json` present | GPU acceleration available (to us or to Spirula) | verified |
| E6 | `VIRTUAL_ENV=/store/shuvam/.venv` mismatches project `.venv` | Warning on every `uv` call | verified |
| E7 | `doctor`: colmap ✗, exiftool ✗, all 9 optional extras ✗ | Core path runs CPU-only today | verified |

## Reversals

| ID | What changed | From → To | Why |
| --- | --- | --- | --- |
| R1-a | I inferred "no NVIDIA Vulkan" from `ldconfig -p` showing only mesa/radeon/lvp ICDs | "Spirula would run on lavapipe (CPU)" → **"NVIDIA Vulkan IS available"** | Vulkan resolves ICDs from JSON manifests in `/usr/share/vulkan/icd.d/`, where `nvidia_icd.json` exists. `ldconfig` is the wrong oracle. Hypothesis **retracted** |
| R2-a | Assumed `uv init` needed to be run for setup | "run uv init + uv sync" → **"uv sync only"** | Isolated `/tmp` probe proved `uv init` refuses and is a no-op. Avoided a blind, potentially destructive command |
| R3-a | Diagnosed the `model_converter` TXT failure as "COLMAP 4 rig/frames.bin incompatibility" | "need a legacy export path" → **"need `text_dir.mkdir()`"** | The error's *first* line said `Directory ... does not exist`. Reading the stack tail sent me the wrong way. **Retracted** |
| R4-a | F6 depth polarity measured on a synthetic 2-tone image | "docstring is correct" → **"docstring is wrong (1 = nearest)"** | The synthetic fixture was out of distribution and produced the opposite verdict to a real photograph. **Retracted** |
| R5-a | Claimed the `write_ply` colour block was "probably fine, just unused" | → **"silent data corruption of every vertex past the first"** | A round-trip test comparing *values* (not just shape/dtype) exposed it |
| R6-a | Called the remaining work "none actionable in code" | → **"3 real gaps existed"** (`shell.py` 38%, PLY ASCII, criterion-6 measurement) | Re-challenging my own conclusion found them. "Nothing left" is a hypothesis, not a fact |

## Open Qs

| ID | Question | Prio | Status |
| --- | --- | --- | --- |
| Q2 | Is real telemetry available for the sample clip, or must GPS be synthesised? | **High** | **ANSWERED: none exists** (ffprobe verified). Synthesised via `tools/synthesize_telemetry.py` — a fixture, not evidence |
| Q3 | Install COLMAP — apt (3.7-2) or conda-forge (4.2.0)? | High | **RESOLVED**: conda-forge 4.2.0 CUDA at `.tools/colmap-env` (apt needs sudo; base-env install would break the uv venv) |
| Q4 | Does Spirula **write** COLMAP-format sparse output? | Low | **[UNKNOWN]** — deferred under D3, only matters if D3 triggers |
| Q5 | Repair the `VIRTUAL_ENV` mismatch warning? | Low | resolved enough — unset in-shell; no repo change warranted |
| Q6 | What is the reference data for criteria 1, 2, 3, 6? | **High** | **OPEN — the only blocker.** Needs a real flight log or NTRO reference cloud |

## Learned

- Spirula's docs index has **no CLI reference file**; the authoritative flag table is
  `src/config/TrainConfig.h` (X-macro driven — CLI, `--help`, GUI and `config.json`
  all expand from one row list). Fetch that before writing any integration.
- The repo retired its Python halves after parity gates — a useful pattern: prove a
  port equals the original with golden-value tests, then delete.
- `uv init` is safe-by-refusal: it checks for `pyproject.toml` before writing anything.
  Still probe destructive-looking commands in `/tmp` first.
