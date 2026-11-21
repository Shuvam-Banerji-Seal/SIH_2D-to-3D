# 01 — Third-party components

What the project builds on, under which licence, and how each component is integrated. The
pipeline's own dependency table is in [`docs/architecture.md`](../docs/architecture.md); this file
keeps the licence reasoning and the evaluation evidence behind it. Last revised 2026-11-22.

## Register

The repository is **GPL-2.0-only**. A GPL-3.0 component therefore runs as a separate program and is
never linked or bundled. Components marked *non-commercial* restrict use of the corresponding output
and are called out in the README.

| Component | Role | Licence | Integration | Outcome |
|---|---|---|---|---|
| spirula-studio | SfM, 3DGS training, meshing (accurate profile) | GPL-3.0 | separate program, CLI (`splat/spirula.py`); built from the submodule | adopted, see R1 |
| Marigold v2 | depth prior (accurate profile) | code and weights Apache-2.0; base model Qwen-Image-Edit-2509 has its own licence | imported from the `third_party/marigold-v2` submodule | adopted; held-out AbsRel 2.7 % after per-image monotone calibration |
| Depth Anything V2 Large | monocular fill (fast profile) | **CC-BY-NC-4.0, non-commercial**; the Small model is Apache-2.0 | `transformers`, held by the engine's model cache | adopted; Small selectable with `dense.mono_model` |
| MoGe-3 ViT-L | optional monocular prior | MIT | `tools/setup_third_party.sh` | optional: half the far-field extrapolation error of Depth Anything, about seven times the GPU time |
| RoMa v2 | dense matching that merges passes | MIT | `tools/setup_third_party.sh` | adopted |
| torchvision RAFT | optical flow | BSD | patched forward, CUDA Graphs (`keyframes/flow.py`) | adopted; 87 → 110 flows/s on an A100, output identical to stock |
| pycolmap (COLMAP) | global mapper on flow tracks | BSD-3-Clause | library | adopted |
| Open3D | GPU TSDF fusion | MIT | library, `mesh` extra | adopted |
| gsplat | fly-through rasteriser | Apache-2.0 | JIT CUDA extension, `render` extra | adopted |
| TRELLIS.2 | generated object, on request | MIT | own environment, subprocess (`tools/setup_trellis2.sh`) | adopted as optional, always labelled *generated* |
| RMBG-2.0, DINOv3 | background removal, backbone of the generator | Bria licence (*non-commercial*); Meta's DINOv3 licence | weights used by TRELLIS.2 | as above |
| Fire3D | complete every object of an RGB-D video | MIT | `tools/fire3d_export.py` | **evaluated, not adopted**, see R2 |
| three.js 0.180.0, Spark | web viewer and splat rendering | MIT | vendored in `viewer/static/vendor/` (three.js files unmodified) | adopted; Spark imports `three/addons/postprocessing/Pass.js`, so that file is required |
| javascript-animation-skills | promo and explainer films | MIT | submodule, used only under `promo/` | adopted for `promo/` |

## R1 — spirula-studio

Evaluated 2026-09-26 from its README, licence file, source tree and documentation index.

| Finding | Source |
|---|---|
| C++ 3D Gaussian Splatting trainer in one binary: built-in SfM, frame extraction, SAM masking, video → splat → textured mesh | README |
| Licence GNU GPL-3.0 | `LICENSE` |
| Mesh writers for PLY, OBJ (+ mtl), glTF, GLB and STL; vertex colour or texture atlas | `src/mesh/MeshExport.h` |
| Reads COLMAP, Nerfstudio and Metashape dataset layouts | `docs/architecture.md` |
| Trains 10 M SH-3 Gaussians in 8 GB of VRAM; Vulkan runs on NVIDIA, AMD, Intel and Apple | README |
| Maintained almost entirely by one person | README |

Consequences: it is a binary, not a Python package, so `uv` cannot install it; and GPL-3.0 against
GPL-2.0-only allows invocation as a separate program but not bundling. Both hold today.

**Outcome.** The first reading (2026-09-26) was not to add it, because the problem statement does not
ask for splats and COLMAP was the cheaper SfM. On 2026-09-27 the pipeline was rebuilt around the GPU
stages and the COLMAP command-line backends were dropped; spirula-studio became the accurate
profile's SfM, trainer and mesher, run through its CLI (`splat/spirula.py`, a watchdog kills a trainer
that stops writing: `test_spirula_watchdog`). The fast profile, which is the deliverable, does not
use it. Its open question, whether it *writes* a COLMAP sparse model, is answered: it does
(`dataset/sparse/N/`). Build: `bash third_party/spirula-studio/build_develop.bash
-DSS_BACKEND=vulkan -DSS_BUILD_GUI=OFF`.

## R2 — Fire3D

Completes every object of a posed RGB-D video from all frames at once. Run on the highrise orbit
(62 GB of weights, 10.6 min), its perception found 17 objects and generated each as an indoor
piece: the tower came out as a hollow frame among furniture-like blocks. It is trained on room
scans and does not carry over to drone footage. The complete model (`complete.py`) does not use it;
`tools/fire3d_export.py` and `tools/fire3d_outdoor_protocol.json` remain so the experiment can be
repeated.

## R3 — Environment

| Item | Value |
|---|---|
| Python | ≥ 3.12; developed on 3.14 (`.python-version`) |
| PyTorch | 2.14 with CUDA 13.2 from the `pytorch-cu132` index; torchvision 0.29 (`uv.lock`) |
| GPU | one NVIDIA A100 80 GB, driver 610; the Vulkan backend of spirula-studio needs a working Vulkan driver |
| ffmpeg | ≥ 5 with NVDEC (`.tools/ffmpeg` or `$DRONE3D_FFMPEG`); `drone3d doctor` reports it |
| Weights | Hugging Face and torch caches on `/store/huggingface` when that disk exists, else the libraries' defaults (`utils/model_cache.py`); Marigold assets in `$DEPTH_ASSETS_DIR` |

## Removed

| Item | Why |
|---|---|
| `third_party/pdoom-video` submodule | Referenced by nothing |
| `docs` dependency group (MkDocs) | There is no MkDocs site |
