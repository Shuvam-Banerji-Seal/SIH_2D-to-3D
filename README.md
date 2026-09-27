# Single-Pass Drone Video → Complete 3D Model, on one GPU

[![License: GPL v2](https://img.shields.io/badge/license-GPL--2.0--only-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](.python-version)
[![uv](https://img.shields.io/badge/managed%20with-uv-5c2d91.svg)](https://docs.astral.sh/uv/)
[![SIH PS](https://img.shields.io/badge/SIH%20PS-26158-orange.svg)](docs/problem-statement.md)

One drone video in — one pass, no flight planning, no ground control — and out
come a **textured 3D mesh** and a **dense point cloud** of the visible scene,
**georeferenced and metric** when a GPS log is supplied, in the formats Smart
India Hackathon problem statement **26158** (NTRO) asks for: **OBJ, PLY, LAS,
GeoTIFF, glTF/GLB and FBX**, with a **web viewer** that measures distances.

Everything runs on one GPU: NVDEC decoding, RAFT optical flow, structure from
motion built on that flow instead of SIFT, flow triangulation completed by a
monocular depth prior, GPU TSDF fusion and GPU texture baking.

## Two profiles

| | `configs/fast.yaml` — the deliverable | `configs/accurate.yaml` / `default.yaml` |
|---|---|---|
| Output | textured mesh + point cloud, DSM + orthophoto | 3D Gaussian Splatting model + mesh from it |
| SfM | RAFT flow tracks → pycolmap global mapper | spirula-studio GPU SfM (SIFT) |
| Geometry | flow triangulation + Depth Anything V2 fill → GPU TSDF | depth-supervised 3DGS (Marigold v2 prior), spirula mesher |
| Time | within or near the official budget (below) | hours per clip |
| Needs | this repository, ffmpeg with NVDEC | + spirula-studio build, Marigold v2 weights |

## How the fast profile works

```mermaid
flowchart LR
    V[drone video] --> K[keyframes<br/>one NVDEC decode, RAFT]
    K -->|passes, overlap-band keyframes,<br/>'is it 3D?' verdict| S[sfm<br/>flow tracks + global mapper]
    S -->|poses| D[dense<br/>flow triangulation + mono fill<br/>GPU TSDF]
    T[GPS log] --> G[georef<br/>ground-levelled 4-DoF]
    S --> G
    D --> E[export<br/>OBJ PLY LAS GeoTIFF GLB FBX<br/>GPU texture, web viewer]
    G --> E
    E --> R[metrics + report.html]
```

1. **One decode.** NVDEC decodes the video once; the same ffmpeg process
   delivers 640 px analysis frames and 1920 px keyframe candidates (stacked in
   one frame), colour conversion runs on the GPU, candidates are nvJPEG-encoded
   as they stream. A 12-pair motion probe lowers the analysis rate for slow
   footage (the sample videos move 0.5–3 px per step at 12 fps).
2. **Passes and keyframes by overlap, not by clock.** Cuts, fades and dissolves
   split edited footage into passes (flow forward–backward consistency collapses);
   letterbox bars are cropped. A grid chained through the flow from keyframe *K*
   picks the next keyframe as the sharpest frame whose co-visibility with *K*
   lies in [0.75, 0.85] — similar enough to match, different enough to see depth.
3. **Is it really 3D?** Residual parallax after the best homography, measured
   with direct keyframe-pair flow and normalised by the noise floor, flags a
   drone yawing on the spot or a planar/far-away scene as `degenerate`.
4. **SfM from flow.** Direct RAFT flow from each keyframe to its next three gives
   sub-pixel multi-view tracks (each step checked against the longer direct
   flows, so drift cannot accumulate); pycolmap's global mapper (GLOMAP) maps
   them. No SIFT: on the Jal Mahal orbit the poses agree with SIFT SfM to 0.2 %
   of the flight extent and 0.28° in rotation.
5. **Dense geometry.** With the poses, every consistent flow vector to
   neighbours 2–12 keyframes away is triangulated (a single pass is 0.1–1°
   between nearby keyframes, so wide baselines matter); Depth Anything V2,
   calibrated per image to that triangulated depth, fills what flow cannot see
   (far field, water) — never the sky (zero disparity). A GPU TSDF with a wide
   truncation band fuses all views into a coloured mesh and point cloud.
6. **Georeferencing.** A single pass is nearly a straight line, so aligning
   camera centres to GPS leaves the roll about it undetermined; the model is
   levelled on its RANSAC ground plane and only yaw, scale and translation are
   fitted. LAS and GeoTIFF are written in UTM with the EPSG code.
7. **Export.** Full-density PLY; a viewable copy (≤ 600k triangles) as GLB,
   textured OBJ/GLB (atlas baked on the GPU from the 1920 px keyframes: each
   triangle from the view that sees it best, unoccluded) and FBX; LAS; DSM and
   orthophoto GeoTIFFs; a self-contained three.js viewer.

## Measured

Shared A100 80 GB (another user's job held ~40 GB and most of the GPU
throughout), budget = 1.5 × video length (the problem statement's 15 minutes
for a 10-minute video). *Completeness* is the share of each registered
keyframe's non-sky pixels whose ray hits the reconstructed mesh
([`experiments/eval_completeness.py`](experiments/eval_completeness.py)).

| Video | Length | Keyframes | Registered | Completeness | Processing | Budget |
|---|---|---|---|---|---|---|
| Qutub Minar, New Delhi | 187 s | 113 | 106 | 0.75 | **243 s** | 281 s ✓ |
| Jal Mahal, Jaipur (edited cinematic, lake) | 55 s | 128 | 128 | 0.78 | 147 s | 82 s ✗ |

The trade-off is set in [`configs/fast.yaml`](configs/fast.yaml): analysing at
a fixed 12 fps gives 0.81 / 307 s and 0.82 / 262 s respectively.

Georeferencing, with synthetic GPS (1.5 m horizontal / 3 m vertical noise) on
the real Jal Mahal models ([`experiments/georef_e2e.py`](experiments/georef_e2e.py)):
rotation error 0.02–0.36°, scale error ≤ 0.5 %, mesh error **0.43 m and 0.86 m
(median) within 100 m of the flight track**, growing with distance (7–18 m for
geometry 300 m+ away across the lake). None of the public sample videos has a
flight log; a real one has not been tested yet.

## Quickstart

Requires an NVIDIA GPU (tested on A100, driver 610 / CUDA 13.x) and `uv`.

```bash
git clone --recurse-submodules https://github.com/Shuvam-Banerji-Seal/SIH_2D-to-3D
cd SIH_2D-to-3D
uv sync --all-extras                       # Python 3.14, torch 2.14 + CUDA 13.2
# a static ffmpeg >= 5 with NVDEC is picked up from .tools/ffmpeg (or $DRONE3D_FFMPEG)
# optional, for FBX: mamba create -p .tools/assimp -c conda-forge assimp && tools/build_meshconv.sh
uv run drone3d doctor

uv run drone3d run --config configs/fast.yaml \
    --set ingest.video=/data/pass.mp4 \
    --set ingest.telemetry=/data/pass.SRT      # optional: DJI SRT, CSV, GPX or JSON
uv run drone3d view outputs/fast               # http://127.0.0.1:8765
```

RAFT and Depth Anything V2 weights download on first use (set `HF_HUB_CACHE` /
`TORCH_HOME` to choose where). Any key can be overridden with
`--set section.key=value`; any stage re-run on an existing run directory with
`--run-dir R --stages dense,export`. Sample footage: `tools/fetch_datasets.sh`
downloads the fifteen public clips listed in [`resources.md`](resources.md).

## Outputs (fast profile)

```
outputs/<run>/
├── export/
│   ├── index.html, scene.json, vendor/   web viewer (drone3d view <run>)
│   └── model_N/
│       ├── mesh.ply                      full-density mesh, vertex colours
│       ├── mesh.glb                      viewable copy (<= 600k triangles)
│       ├── mesh_textured.{glb,obj,mtl}   textured from the keyframes (+ _albedo.jpg)
│       ├── mesh.fbx                      textured, via assimp
│       ├── points.{ply,las}              dense point cloud (LAS in UTM + EPSG when georeferenced)
│       └── dsm.tif, ortho.tif            GeoTIFF surface model and orthophoto
├── dataset/images/pass_NN/*.jpg          keyframes (1920 px)
├── dataset/sparse/N/                     COLMAP models, one per pass
├── georef/                               transforms, ENU sparse clouds, camera_track.geojson
├── keyframes/keyframe_timeline.png       passes, overlap and verdicts along the video
├── */result.json, */gpu_timeline.json    per-stage results and GPU telemetry
└── metrics/metrics.json                  incl. "processing": seconds per stage vs the budget
```

## The accurate profile (3D Gaussian Splatting)

`configs/default.yaml` keeps the first pipeline: spirula-studio GPU SfM (213/213
Jal Mahal keyframes, 0.79 px), Marigold v2 depth calibrated per image to the SfM
tie points (monotone map; held-out AbsRel 2.7 %), depth-supervised 3DGS (held-out
PSNR 35.0 dB on the largest model), meshes from the splats and gsplat fly-through
renders. It needs the spirula-studio build and the Marigold v2 / Qwen-Image-Edit
weights (see [`docs/architecture.md`](docs/architecture.md)) and takes hours per
clip. The method and its experiments are in [`paper/`](paper/).

## Repository

```
src/drone3d/
├── io/nvdec.py         NVDEC decode (analysis + keyframe candidates in one pass), GPU colour, nvJPEG
├── io/telemetry.py     CSV / DJI SRT / GPX / JSON flight logs
├── keyframes/          RAFT flow (CUDA Graphs), pass segmentation, overlap-band selection, 3D verdict
├── fastsfm/            flow tracks, pycolmap mapping, flow triangulation, mono fill, TSDF (fast profile)
├── export/             OBJ/PLY/GLB/FBX/LAS/GeoTIFF writers, GPU texture baking
├── viewer/static/      three.js viewer (vendored, MIT)
├── depth/, splat/      Marigold v2 and spirula-studio wrappers (accurate profile)
├── geo/                WGS84/ENU, Umeyama, ground-levelled 4-DoF georeferencing
├── gpu/monitor.py      NVML sampler (records GPU sharing)
├── pipeline.py, config.py, cli.py
experiments/            scripts behind every number above and in the paper
paper/, promo/          LaTeX paper; code-drawn promo film + compositor
third_party/            spirula-studio, marigold-v2, javascript-animation-skills (submodules)
tools/                  dataset download, synthetic control clip, FBX converter
```

## Limitations

Timings are from a shared GPU; Jal Mahal (an edited clip with five camera moves
and a lake that defeats optical flow) is 1.8× over its budget. Completeness is
~0.75 of what the camera saw: surfaces the single pass never faced cannot be
reconstructed, and flow fails on water and sky. Georeferencing is verified with
synthetic GPS only. Dynamic objects are not masked.

## Licences

This repository is GPL-2.0-only. spirula-studio (GPL-3.0) is used as a separate
program through its command line; three.js is MIT. **Depth Anything V2 Large,
the fast profile's default depth prior, is CC-BY-NC-4.0 (non-commercial)**; the
Small model is Apache-2.0 (`--set dense.mono_model=depth-anything/Depth-Anything-V2-Small-hf`).
Marigold v2 weights are Apache-2.0 on top of Qwen-Image-Edit-2509 (its own
licence). The sample videos are third-party YouTube content used for research
only and are not redistributed.
