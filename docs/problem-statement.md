# Problem Statement — Single-Pass Drone Video to Accurate 3D Model Generation System

| Field | Value |
| --- | --- |
| Problem Statement ID | 26158 |
| Title | Single-Pass Drone Video to Accurate 3D Model Generation System |
| Organisation | National Technical Research Organisation (NTRO) |
| Department | National Technical Research Organisation (NTRO) |
| Category | Software |
| Theme | Robotics and Drones |
| Dataset | Provided in real time (additional details shared by NTRO) |

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

| # | Challenge | Where it is addressed in this repo |
| --- | --- | --- |
| 1 | Limited viewing angles from one flight path | Multi-stage SfM + dense MVS, monocular depth prior for occlusions |
| 2 | Motion blur and video compression artifacts | Quality scoring + keyframe selection, Wiener/unsharp deblur (`preprocess/deblur.py`) |
| 3 | Variable illumination and shadows | Exposure-aware frame scoring, exposure clipping masks, shadow suppression in dynamic masking |
| 4 | Dynamic objects (vehicles, humans, animals) | Motion/background-subtraction masks, optional YOLO semantic masks (`preprocess/dynamic.py`) |
| 5 | GPS inaccuracies and sensor noise | Robust similarity fit (Umeyama) on camera centers, RMSE reporting (`geo/georef.py`) |
| 6 | Real-time / near-real-time requirements | `configs/fast.yaml` profile, streaming sampling and tiled outputs |
| 7 | Reconstruction of occluded surfaces | Monocular depth completion + MVS fusion (`dense/`) |
| 8 | Metric accuracy without extensive GCPs | GPS/IMU-prior alignment, scale estimation, optional reference-cloud metrics |

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

## Desired output

| # | Output | Format | Definition of done |
| --- | --- | --- | --- |
| 1 | Sparse reconstruction | COLMAP model + `sparse.ply` | Camera poses and tie points registered for the majority of keyframes |
| 2 | Dense point cloud | `fused.ply` / georeferenced PLY | Metric scale, colour per point, coverage of terrain, structures and vegetation |
| 3 | Textured 3D mesh | `mesh.obj` + textures (`mesh.ply`, `mesh.glb`) | Watertight-enough surface with facade/roof/road detail suitable for visualization |
| 4 | Georeferenced outputs | PLY in local ENU + `camera_track.geojson` | Horizontal/vertical RMSE reported against GPS; EPSG-tagged |
| 5 | Measurement-ready artifacts | bounds/extent, voxel coverage, scale check | Extents and scale error reported in `metrics.json` |
| 6 | Run report | `report.html`, `manifest.json` | Stage statuses, metrics, artifacts and config reproducible in one page |

## Evaluation criteria

| # | Criterion | Metric | Target (indicative) |
| --- | --- | --- | --- |
| 1 | Geometric accuracy | RMSE vs. reference / GPS checkpoints | ≤ 1 × GSD horizontal, ≤ 2 × GSD vertical |
| 2 | Metric scale correctness | Relative scale error | ≤ 2 % without GCPs |
| 3 | Completeness | Fraction of reference points reconstructed | ≥ 80 % at 0.5 m threshold |
| 4 | Visual quality | Texture quality, absence of holes/holes filled | Facades and rooftops recognizable |
| 5 | Dynamic-object handling | Ghost artifacts in final model | No visible vehicle/person ghosting |
| 6 | Georeferencing quality | Horizontal/vertical GPS RMSE | ≤ 3 m horizontal with consumer GPS |
| 7 | Robustness | Successful runs on blurry / low-light passes | No crash, graceful degradation reported |
| 8 | Latency | End-to-end processing time | Near-real-time on `fast` profile; batch on `accurate` |
| 9 | Usability | One-command run + HTML report | `drone3d run --config ...` produces complete run directory |
| 10 | Reproducibility | Same input + config → same metrics | Deterministic stage outputs, versioned `uv.lock` |

## Potential applications

Border and strategic area mapping, disaster damage assessment, urban planning
and smart cities, infrastructure inspection, construction progress monitoring,
archaeological documentation, digital twin generation, military reconnaissance
and mission planning.
