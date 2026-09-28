# Problem Statement — Single-Pass Drone Video to Accurate 3D Model Generation System

| Field | Value |
| --- | --- |
| Problem Statement ID | 26158 |
| Title | Single-Pass Drone Video to Accurate 3D Model Generation System |
| Organisation | National Technical Research Organisation (NTRO) |
| Department | National Technical Research Organisation (NTRO) |
| Category | Software |
| Theme | Robotics and Drones |
| Dataset | Provided in real time at the event (no video link in the statement) |

## Background

Generation of accurate 3D models of buildings, infrastructure, terrain and
objects typically requires multiple drone passes, extensive image overlap,
specialized flight planning and significant post-processing time. In
operational scenarios such as disaster response, surveillance, infrastructure
inspection, military reconnaissance and rapid mapping, there is often only a
single opportunity to capture data over the target area.

A solution capable of generating an accurate, textured 3D model from a
single-pass drone video would significantly reduce mission time, operator
effort, data acquisition requirements and processing complexity, while enabling
near real-time situational awareness.

## Objective

Design and develop an AI-enabled system that generates a georeferenced and
metrically accurate 3D model of a scene using **only a single-pass drone video
stream** captured from a moving UAV. The system must reconstruct:

1. 3D terrain and structures
2. Building facades and rooftops
3. Roads and infrastructure
4. Vegetation and obstacles
5. Textured 3D meshes or point clouds

The generated model must be suitable for visualization, measurement and
analysis.

## Key challenges

| # | Challenge | Where it is addressed | Status |
|---|---|---|---|
| 1 | Limited viewing angles from one flight path | Overlap-band keyframes (~1/(1−τ) views per point); per-pass "is it 3D?" parallax test; flow triangulation against neighbours up to 12 keyframes away (single-pass baselines are 0.1–1°); Depth Anything V2 fill calibrated to the triangulated depth (`keyframes/`, `fastsfm/`) | implemented, evaluated (completeness: median 0.88 over the 15 sample videos, 0.48–0.95) |
| 2 | Motion blur and compression artefacts | Keyframe choice prefers the sharpest frame inside the overlap band (Laplacian variance vs a rolling median) | implemented; no explicit deblurring |
| 3 | Variable illumination and shadows | Fades trimmed from passes; per-triangle best-view texturing with exposure gain compensation across views (OpenCV 5, fast profile); bilateral-grid exposure correction in 3DGS training (accurate profile) | implemented |
| 4 | Dynamic objects | Not handled explicitly beyond the photometric robustness of training; spirula-studio's `--distraction-robustness` and SAM masking are available but not wired in | **open** |
| 5 | GPS inaccuracies and sensor noise | Ground-levelled 4-DoF alignment, leave-one-out error, jackknife scale uncertainty (`geo/georef.py`) | implemented; evaluated in simulation (sample videos have no GPS) |
| 6 | Real-time / near-real-time | Fast profile: one NVDEC decode, motion-adaptive analysis rate, SfM from optical flow (no SIFT) with passes mapped while the next is tracked, GPU TSDF and texture baking; a warm engine keeps the networks loaded; **live mode** records RTSP/RTMP/SRT/UDP/HLS or a camera in segments and models each one while the next is recorded (`engine/`) | see the benchmark table in the README (every sample video, warm engine); live: a model 26–38 s after each 30 s segment closes |
| 7 | Occluded surfaces | Monocular depth fill where flow cannot triangulate; TSDF with a wide truncation band | partial: surfaces the pass never faced are absent |
| 8 | Metric accuracy without GCPs | Ground-levelled GPS fit with leave-one-out error and jackknife scale uncertainty | synthetic GPS on real models: scale error ≤ 0.5 %, mesh error < 1 m within 100 m of the track; needs a real flight log |

## Input data

**Mandatory**

| Input | Format | Location in config |
| --- | --- | --- |
| Drone video (1080p / 4K) | `.mp4`, `.mov`, `.mkv`, `.avi` | `ingest.video` |
| GPS coordinates | `.csv`, `.srt`, `.gpx`, `.json` | `ingest.telemetry` |
| Flight metadata | telemetry rows/columns | `ingest.telemetry` |

**Optional**

| Input | Format | Location in config |
| --- | --- | --- |
| IMU (pitch/roll/yaw) | telemetry columns | `ingest.telemetry` |
| Barometric altitude | telemetry column (`rel_alt`, `altitude`) | `ingest.telemetry` |
| Camera intrinsics | pinhole `fx, fy, cx, cy` | `CameraIntrinsics` / camera-model config |
| RTK/PPK corrections | high-accuracy GPS columns | `geo.*` |

## Desired output (official)

From the NTRO problem-statement document (SIH 2026, "Problem Statement – 17",
linked in [`resources.md`](../resources.md)):

| Parameter | Target | This repository today |
| --- | --- | --- |
| Reconstruction type | 3D mesh / point cloud | Textured mesh + dense point cloud (fast profile); 3DGS + mesh (accurate profile) |
| Processing time | **< 15 minutes for a 10-minute video** | Fast profile on one A100: 0.64 s per video second over the 11 survey-style sample videos (budget 1.5), 11 of 15 within the budget; the misses are FPV flights and clips under half a minute (README, Measured) |
| Spatial accuracy | ≤ 1 m | < 1 m within 100 m of the flight track with 1.5 m GPS noise (synthetic GPS on real models, `experiments/georef_e2e.py`); error grows with distance (7–18 m at 300 m+) |
| Coverage | Entire visible scene | Measured as the share of every registered keyframe's non-sky pixels the mesh covers: median 0.88 over the 15 sample videos; lowest where the far field lies beyond three times the triangulated range (Cristo Redentor 0.49) |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx | All: OBJ (textured), PLY, LAS (UTM + EPSG), GeoTIFF (DSM + orthophoto), GLB (y-up, as glTF requires), FBX; also STL, a Blender scene (.blend, texture packed) and web Gaussian splats |
| Visualization | Web-based or desktop viewer | Web console (`drone3d ui`): engine and model control, GPU telemetry, every option, live sessions, and an explorer with every layer as a switch -- textured / shaded / wireframe mesh, dense cloud, Gaussian splats, flight path, keyframe photos, fused depth maps, the source video -- orbit, fly and follow-flight navigation, look-through-keyframe, measuring, screenshots; the same explorer ships with every export; a file viewer opens any GLB, OBJ, PLY, STL or FBX on its own. Blender opens the .blend directly. |

## Evaluation criteria (official weights)

| Criterion | Weight |
| --- | --- |
| Reconstruction accuracy | 30 % |
| Model completeness | 20 % |
| Processing speed | 20 % |
| Innovation | 15 % |
| Scalability | 10 % |
| User interface | 5 % |

An earlier version of this file listed numeric targets (GSD multiples, 2 %
scale error, 80 % completeness at 0.5 m, ...). They are not in the official
document and have been removed.

## Potential applications

Border and strategic area mapping, disaster damage assessment, urban planning
and smart cities, infrastructure inspection, construction progress monitoring,
archaeological documentation, digital twin generation, military reconnaissance
and mission planning.
