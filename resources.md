# Resources

Reference material for the single-pass drone → 3D pipeline: sample footage,
problem-statement links and tooling. Videos are downloaded locally under
`datasets/` (git-ignored, ~2.1 GB) purely for testing and benchmarking.

## Sample drone footage

All clips are third-party YouTube content used for research/testing only — the
repository stores links and local paths, not the media itself. Extracted frames
(if any) live under `frames/<video title>/`, also git-ignored.

| # | Title | Link | Local file | Size | Reconstruction notes |
| --- | --- | --- | --- | --- | --- |
| 1 | The Messiah \| Cristo Redentor 4K Drone Footage | https://www.youtube.com/watch?v=OO4cuHmkm3k | `datasets/The Messiah ｜ Cristo Redentor 4K Drone Footage [OO4cuHmkm3k].webm` | 78 MB | Landmark + terrain, good facade/occlusion case |
| 2 | Qutub Minar, New Delhi, India 4K Drone Video | https://www.youtube.com/watch?v=47YrmPhMeho | `datasets/Qutub Minar, New Delhi, India 4K Drone Video ｜ UNESCO World Heritage Site [47YrmPhMeho].webm` | 174 MB | Tall structure, strong verticals, Indian site |
| 3 | Eiffel Tower Drone 4k | https://www.youtube.com/watch?v=Qx_c1X3zfEc | `datasets/Eiffel Tower Drone 4k [Qx_c1X3zfEc].webm` | 487 MB | Fine lattice detail, texture stress test |
| 4 | Notre Dame Drone Paris 4k | https://www.youtube.com/watch?v=4V1ZGEKQE2A | `datasets/Notre Dame Drone Paris 4k [4V1ZGEKQE2A].webm` | 384 MB | Facades, rooftops, complex geometry |
| 5 | Colosseum like never before: 4K drone aerial view | https://www.youtube.com/watch?v=wcl0k1WPat0 | `datasets/Colosseum like never before： 4K drone aerial view [wcl0k1WPat0].webm` | 223 MB | Curved structure, occluded interior |
| 6 | ANGKOR WAT Breathtaking • Cinematic Aerial Adventure in 4K | https://www.youtube.com/watch?v=by8H_fLpJpg | `datasets/ANGKOR WAT Breathtaking • Cinematic Aerial Adventure in 4K [by8H_fLpJpg].webm` | 296 MB | Temple complex + vegetation + water |
| 7 | Reichstag Berlin in 4K \| Stunning Drone Views | https://www.youtube.com/watch?v=wXWNsayEP90 | `datasets/Reichstag Berlin 🇩🇪 in 4K ｜ Stunning Drone Views of Germany’s Parliament [wXWNsayEP90].webm` | 240 MB | Domes, facades, urban context |
| 8 | Petronas Twin Tower in FPV drone - 4K | https://www.youtube.com/watch?v=p9jSKslwZWo | `datasets/Petronas Twin Tower in FPV drone - 4K - Niche Films [p9jSKslwZWo].webm` | 103 MB | Aggressive FPV motion, motion-blur case |
| 9 | Above clouds - PNB Merdeka 118 \| Cinematic Aerial Shots | https://www.youtube.com/watch?v=ucpkDTp1Th4 | `datasets/Above clouds - PNB Merdeka 118 ｜ Cinematic Aerial Shots [ucpkDTp1Th4].webm` | 98 MB | Skyscraper, clouds/haze, dynamic range |
| 10 | Aerial Views of Rural Riches: Drone Shot of Farmland | https://www.youtube.com/watch?v=p8eRmxosalI | `datasets/Aerial Views of Rural Riches： Drone Shot of Farmland 🌾🚁 [p8eRmxosalI].webm` | 22 MB | Terrain/vegetation, low-texture fields |

Total footage: ~2.1 GB, all `.webm` (VP8/VP9 in WebM containers).

## Problem statement links

- SIH 2026 problem statement 26158 (NTRO) additional information:
  https://drive.google.com/file/d/119hjXkLhMW_AhQ4cyYz-XJgcVz4BA-hD/view
- Problem statement summary and requirement mapping:
  [`docs/problem-statement.md`](docs/problem-statement.md)

## Tooling and references

| Resource | Link | Used for |
| --- | --- | --- |
| COLMAP | https://colmap.github.io/ | SfM, MVS, meshing, texturing |
| Open3D | https://www.open3d.org/ | Poisson meshing, point-cloud processing |
| Trimesh | https://trimesh.org/ | Mesh export/decimation (OBJ, GLB, STL) |
| Depth Anything V2 | https://github.com/DepthAnything/Depth-Anything-V2 | Monocular depth prior for occlusions |
| Ultralytics YOLO | https://docs.ultralytics.com/ | Dynamic-object (vehicle/person) masking |
| uv | https://docs.astral.sh/uv/ | Python environment and lockfile management |

## Usage

```bash
# Inspect a clip
ffprobe "datasets/Eiffel Tower Drone 4k [Qx_c1X3zfEc].webm"

# Run the pipeline on one pass
uv run drone3d run \
  --config configs/fast.yaml \
  --set ingest.video="datasets/Eiffel Tower Drone 4k [Qx_c1X3zfEc].webm"
```

These clips have no GPS sidecar files, so georeferencing (`georef`) will be
skipped; supply a `.srt`/`.csv`/`.gpx`/`.json` telemetry log via
`ingest.telemetry` to exercise it.
