# Architecture

## Stages and their files

Stages talk to each other only through files in one run directory, so any
stage can be re-run alone (`drone3d run --run-dir R --stages splat,mesh`). Each
stage writes `<stage>/result.json`, and — when it used the GPU —
`<stage>/gpu_timeline.json` with NVML utilisation, power, memory, host RSS and
whether other processes shared the GPU.

| Stage | Reads | Writes | Runs on |
|---|---|---|---|
| `ingest` | video, telemetry | `ingest/result.json`, `ingest/telemetry.json` | CPU (ffprobe) |
| `keyframes` | video | `dataset/images/pass_NN/*.jpg`, `keyframes/{selection,keyframes,result}.json`, timeline figure | NVDEC, RAFT (CUDA Graphs), nvJPEG |
| `sfm` | keyframes | `dataset/sparse/N/` (COLMAP), `sfm/result.json` | spirula-studio (Vulkan compute), or `backend: flow`: RAFT tracks + pycolmap global mapper |
| `dense` (fast) | keyframes, SfM | `dense/model_N/{mesh,points}.ply` | RAFT, flow triangulation, Depth Anything V2, Open3D GPU TSDF |
| `depth` | keyframes, SfM | `dataset/depths/`, `dataset/depth_raw/`, `depth/result.json` | Marigold v2 (NF4 DiT, bf16) |
| `splat` | dataset (+ depths) | `splats/model_N/` (splat.ply, held-out renders, metrics.json) | spirula-studio trainer |
| `mesh` | splats | `splats/model_N/mesh*.{ply,obj,glb}` | spirula-studio mesher |
| `georef` | SfM, keyframe GPS | `georef/*_sparse_enu.ply`, `camera_track.geojson` | CPU |
| `export` (fast) | dense, georef, keyframes | `export/model_N/` (PLY, GLB, textured OBJ/GLB, FBX, LAS, DSM/ortho GeoTIFF, `frame.json`), `export/index.html` viewer | GPU texture baking, CPU writers, assimp |
| `render` | splats, SfM | `render/*_flythrough.mp4` | gsplat + NVENC |
| `metrics`, `report` | everything | `metrics/metrics.json`, `report.html`, `manifest.json` | CPU |

## Key decisions and the evidence behind them

| Decision | Evidence (reproduce with) |
|---|---|
| Keyframes by measured co-visibility, not a fixed rate | Jal Mahal: 1 fps sampling registered 40/55 frames in 2 disconnected models (largest 22); overlap-band keyframes registered 213/213 in one model per group of overlapping passes (`outputs/jal_mahal/sfm/result.json`) |
| Chained flow for overlap, **direct** flow for the 3D test | chained RAFT error grows ~linearly (0.90 px median at 24 frames, 1.12 px at 32) vs 0.12–0.13 px direct (`experiments/flow_drift.py`) |
| Parallax SNR decides "is it 3D?", GRIC only confirms | pure-rotation control: SNR 1.24 (degenerate) while GRIC still preferred F on 60 % of pairs (`outputs/controls/pure_rotation_run`) |
| One focal length per pass (`radial`) | OpenCV's independent fx/fy drifted 10–20 % apart with no reprojection gain (0.761 vs 0.771 px) |
| Letterbox crop, fade trimming | bars inflated overlap and held-out PSNR; a fade-out keyframe scored 13 dB as a held-out view |
| Monotone depth calibration | held-out AbsRel vs SfM 5.0→2.7 %, 7.5→3.3 % over log-affine (`outputs/jal_mahal/depth/result.json`, `cv_*`) |
| Poses from flow tracks, not SIFT (fast) | Jal Mahal pass 3: centres within 0.2 % of the extent and rotations 0.28° of SIFT SfM, 97/97 registered, 16 s of mapping (`experiments/flow_sfm.py`, `paper/figures/fast_flow_sfm.json`) |
| Global mapper, capped tracks (fast) | COLMAP incremental defaults (16° init) register nothing on some passes; GLOMAP with ~12 tracks/image maps the 97-keyframe pass in 11–16 s |
| Wide TSDF truncation band (fast) | band 4 → 12 voxels: completeness 0.47 → 0.80, depth error vs triangulated 0.42 → 0.60 % (`experiments/dense_sweep.py`) |
| Adaptive analysis rate (fast) | Qutub Minar: keyframe stage 186 → 70 s, completeness 0.81 → 0.75; within the budget (`configs/fast.yaml`) |
| GPU triangle-soup texture (fast) | Open3D UVAtlas took 40 s per 82k triangles and crashed on decimated TSDF meshes; soup atlas: seconds, 3-view blend |
| Ground-levelled 4-DoF georeferencing | straight pass: 7-DoF similarity 147–221 m off 100 m from the track, levelled fit 0.2–1.8 m (`experiments/georef_study.py`) |
| RAFT without host syncs + CUDA Graphs | A100, 640×360 pairs: 87 → 110 flows/s (1.27×), output identical to stock (`experiments/bench_gpu.py`, `paper/figures/bench_gpu.json`) |
| Flow batch not raised on a free GPU | idle A100, 480×272: 206, 205, 207 flows/s at 16, 32, 64 pairs per replay; cuDNN refuses 128 (`experiments/bench_raft_batch.py`) |
| Every SfM component of a pass is a model (fast) | Hanoi's one pass mapped as 43 + 25 + 6 keyframes; keeping the largest dropped 31 registered views and triggered a 128 s second mapper |
| Three views per surface voxel, two on small models (fast) | Open3D's default asks for four: +3–12 points of completeness on large models, +8–65 on small ones (a 5-view shot: 0 → 0.65), ≤ 0.1 % more depth error (`experiments/tsdf_weight.py`) |
| Fill stops at 3× the triangulated range (fast) | held-out: the prior is 3–15 % off just past the calibrated range (42 % on the lake model) and 47–87 % beyond 3× (`experiments/far_field.py`) |
| One splat trainer at a time | Jal Mahal's 5 models: 178 s in sequence, 249 s two and 264 s three at once (`experiments/splat_parallel.py`) |

## GPU and memory discipline

- Decoding: ffmpeg ≥ 5 with `-hwaccel cuda` + `scale_cuda`, raw NV12 out, colour
  conversion on the GPU; a background thread streams chunks so decoding
  overlaps RAFT.
- Flow and analysis frames are preallocated on the GPU from the probed frame
  count and released before 4K extraction; 4K conversion runs in 8-frame
  sub-batches.
- Marigold's NF4 base transformer is cached (`$DEPTH_ASSETS_DIR/cache`), so a
  load reads ~11 GB instead of re-quantising 39 GB.
- Host RAM: spirula caches training images on disk by default
  (`splat.cache_images`), extraction holds ≤ 2 groups of 24 raw 4K frames, the
  mesher uses `mesh.num_threads` (12), and every stage records its peak RSS.

## Runtime: engine, console, live footage

```
browser ──HTTP── drone3d ui (FastAPI, no torch) ──HTTP── drone3d engine (one per GPU)
   │                 │  supervises / restarts the engine          │  model cache: RAFT, Depth Anything V2, Marigold
   │                 │  NVML sampler, 1 Hz, 10 min kept            │  job queue (slots), live sessions (ffmpeg recorder)
   └── explorer ◄────┴── /runs/<run>/export/…  ◄───────────────────┘  writes the same run files as `drone3d run`
```

- **Model cache** (`engine/models.py`): stages call `models.raft()`, `models.mono()`,
  `models.marigold()`. Outside the engine these build and free a network per use,
  exactly as before; inside it the network stays on the GPU, one network serves
  every batch size and slot, and RAFT's CUDA graphs are captured once per
  resolution and slot -- the three most recently used shapes are kept, since
  each graph holds a private memory pool (1-1.5 GB at 16 pairs of 480 px) that
  `empty_cache` cannot return. A model loads only if NVML shows its footprint plus
  `--reserve-gb` free; loading happens outside the cache lock; unloading waits
  for the runs using it.
- **Jobs** (`engine/service.py`): `--slots N` runs at once (1 is how a video's
  time is measured); Marigold runs take the engine to themselves; settings that
  scale with memory (TSDF budget, flow batch, texture size) are fitted to a share
  of the free memory at run start. After every job the CUDA context is checked;
  a sticky fault saves the queue and warm list and exits for the console to
  restart it (`.engine_poisoned`); `POST /restart` drains and hands over the
  queue (`.engine_restart`), which is how new code or a new slot count is applied.
- **Live** (`engine/live.py`): ffmpeg records the source (RTSP / RTMP / SRT /
  UDP / HLS / HTTP, a V4L2 camera on NVENC, or a file replayed with `-re`) into
  `segments/seg_NNNN.mkv` at keyframes without re-encoding; every closed segment
  is queued ahead of other work; with a flight log, `ingest.telemetry_offset_s`
  aligns the log to each segment and the ENU origin is pinned to its first fix.
- **Console** (`app/`): `schema.py` turns the config dataclasses and the comments
  beside each field into the options form; `server.py` serves runs, scenes
  (URLs prefixed per run or per live segment), keyframe/depth filmstrips,
  thumbnails, the source video (range requests) and the engine proxy;
  `static/js/` is the front end; `viewer/static/explorer.js` is the 3D explorer
  shared with the standalone viewer in every export.

## Third-party components

| Component | Role | Licence | Integration |
|---|---|---|---|
| spirula-studio | SfM, 3DGS training, meshing | GPL-3.0 | separate program, CLI (`splat/spirula.py`) |
| Marigold v2 | depth prior | code/weights Apache-2.0 (base: Qwen-Image-Edit-2509) | imported from `third_party/marigold-v2` |
| torchvision RAFT | optical flow | BSD-3 | patched forward (`keyframes/flow.py`) |
| gsplat | fly-through rasteriser | Apache-2.0 | JIT CUDA extension |
| javascript-animation-skills | promo film | MIT | `promo/` |
