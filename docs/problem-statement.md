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
| 1 | Limited viewing angles from one flight path | Overlap-band keyframes (~1/(1−τ) views per point); per-pass "is it 3D?" parallax test; depth prior for weakly observed surfaces (`keyframes/`, `depth/`) | implemented, evaluated |
| 2 | Motion blur and compression artefacts | Keyframe choice prefers the sharpest frame inside the overlap band (Laplacian variance vs a rolling median) | implemented; no explicit deblurring |
| 3 | Variable illumination and shadows | spirula-studio's per-image bilateral-grid / PPISP exposure correction during training; fades trimmed from passes | implemented |
| 4 | Dynamic objects | Not handled explicitly beyond the photometric robustness of training; spirula-studio's `--distraction-robustness` and SAM masking are available but not wired in | **open** |
| 5 | GPS inaccuracies and sensor noise | Ground-levelled 4-DoF alignment, leave-one-out error, jackknife scale uncertainty (`geo/georef.py`) | implemented; evaluated in simulation (sample videos have no GPS) |
| 6 | Real-time / near-real-time | GPU end to end; official budget < 15 min per 10-min video | **not met** by the 3DGS profile (hours per 55 s clip); a fast mesh/point-cloud profile is being built |
| 7 | Occluded surfaces | Depth prior supervision; 3DGS/mesh interpolation | partial |
| 8 | Metric accuracy without GCPs | GPS-scaled similarity with reported uncertainty (≤ 0.6 % scale error in simulation) | implemented; needs a real flight log to validate |

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
| Reconstruction type | 3D mesh / point cloud | Sparse SfM cloud; textured mesh extracted from the splat model; no dense point cloud yet |
| Processing time | **< 15 minutes for a 10-minute video** | **Not met.** The 3DGS profile takes hours for a 55 s clip (30k-step training per model, 41 min meshing) |
| Spatial accuracy | ≤ 1 m | Georeferencing implemented and tested in simulation only; the sample videos carry no GPS |
| Coverage | Entire visible scene | Not measured |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx | OBJ, PLY, GLB; **LAS, GeoTIFF, FBX missing** |
| Visualization | Web-based or desktop viewer | **Missing** (HTML run report only) |

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
