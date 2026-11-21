# 00 — Understanding

What the project must do, why it is hard, and the contracts the code relies on. Stage tables and
the evidence behind each design choice are in [`docs/architecture.md`](../docs/architecture.md); the
official statement is in [`docs/problem-statement.md`](../docs/problem-statement.md). Last revised
2026-11-22.

## Problem

Smart India Hackathon problem statement **26158** (NTRO): from **one pass** of a moving UAV, no flight
planning and no ground control, produce a georeferenced, metrically accurate, textured 3D model of
the visible scene as a mesh or point cloud, in OBJ, PLY, LAS, GeoTIFF, glTF/GLB and FBX, viewable in a
web or desktop viewer. The budget is **15 minutes for a 10-minute video**. Scoring: reconstruction
accuracy 30 %, completeness 20 %, processing speed 20 %, innovation 15 %, scalability 10 %, user
interface 5 %.

## Why a single pass is hard

| Difficulty | Consequence | Mechanism in the code |
|---|---|---|
| Thousands of redundant frames, a few bad ones | A fixed sampling rate wastes work where the drone is slow and leaves gaps where it is fast | Keyframes chosen by measured co-visibility: the sharpest frame whose overlap with the last keyframe lies in a band (`keyframes/select.py`) |
| The footage may not contain 3D at all | A drone yawing on the spot, or a planar or distant scene, gives video a homography explains; no computation recovers depth from it | Per-pass verdict from rotation-compensated parallax, measured with *direct* keyframe-pair flow and normalised by the noise floor (`keyframes/geometry.py`) |
| Baselines of 0.1–1° between neighbouring keyframes | Triangulation against the nearest view is ill-conditioned | Flow is triangulated against neighbours up to 12 keyframes away; a monocular prior calibrated to that depth fills what flow cannot see (`fastsfm/`) |
| A single pass is nearly a straight line | Aligning camera centres to GPS leaves the roll about the flight axis undetermined | Model levelled on its ground plane; only yaw, scale and translation fitted (`geo/georef.py`) |
| Time scales with footage length | Per-video-second cost must stay below 1.5 s | One NVDEC decode, SfM from optical flow instead of SIFT, GPU TSDF and texture baking, a warm engine (`io/nvdec.py`, `engine/`) |
| Surfaces the pass never faced | Completeness is bounded by what the camera saw | Reported as view completeness; the optional generated and complete models are labelled as *generated*, never measured (`generate.py`, `complete.py`) |

## Artifact contracts

Stages exchange data only through files in one run directory, so any stage can be re-run alone
(`drone3d run --run-dir R --stages dense,export`). Each stage writes `<stage>/result.json`.

| Contract | Detail | Guarded by |
|---|---|---|
| Stage registry | `ALL_STAGES` in `config.py`; `Pipeline._stage_<name>` implements each | `test_config`, `test_report` |
| COLMAP models | `dataset/sparse/N/{cameras,images,points3D}.bin`, one per pass or merged group of passes | `test_fastsfm_components` |
| Depth maps | 16-bit PNG, linear z-depth scaled so the 99.9th percentile is 65535, 0 = no data | `test_depth_align` |
| PLY | Binary PLY is **interleaved per vertex**; a separate colour block corrupts every vertex after the first | `test_ply` (round trip compares values) |
| Frames | Survey frame is z-up ENU; GLB is y-up as glTF requires; STL keeps the survey frame | `test_export_formats` |
| LAS | LAS 1.4, point format 7 (RGB), millimetre quantisation, CRS as WKT (UTM + EPSG) when georeferenced | `test_export_formats` |
| GeoTIFF | North-up, PixelIsArea, scale and tie point, projected EPSG or user-defined CRS | `test_export_formats` |
| Camera track | `georef/camera_track.geojson` is WGS84 (RFC 7946 has no `crs` member) | — |
| Console help | Each option's help text is the comment on the same source line as its field in `config.py` (`app/schema.py`); **do not reformat `config.py`** | `test_console_api` |

## Constraints

- Licence: this repository is GPL-2.0-only. spirula-studio (GPL-3.0) runs as a separate program
  through its command line and is never linked or bundled. Third-party terms are in
  [`01-research.md`](01-research.md).
- Environment: Python ≥ 3.12 (developed on 3.14), PyTorch 2.14 with CUDA 13.2, one NVIDIA GPU
  (developed on an A100 80 GB). Heavy dependencies sit behind extras and are imported where used;
  everything the code imports is declared in `pyproject.toml` (`tests/test_dependencies.py`).
- Tests are offline and deterministic; those needing a GPU or an optional library skip themselves.

## Open questions

| ID | Question | Status |
|---|---|---|
| Q1 | How accurate is georeferencing on a real flight log? | **Open.** None of the 15 sample videos carries GPS; verified with synthetic GPS only (see [`08`](08-final-verification-report.md)) |
| Q2 | How are dynamic objects (vehicles, people) kept out of the model? | **Open.** Not masked; spirula-studio's `--distraction-robustness` and SAM masking exist but are not wired in |
| Q3 | Can the far field beyond three times the triangulated range be filled? | **Open.** The prior is 47–87 % off there, so it is left empty ([`docs/roadmap.md`](../docs/roadmap.md), item 3) |
| Q4 | Does spirula-studio write a COLMAP sparse model? | **Answered, yes.** `dataset/sparse/N/` is its output ([`docs/data-formats.md`](../docs/data-formats.md)), which `georef` consumes |
