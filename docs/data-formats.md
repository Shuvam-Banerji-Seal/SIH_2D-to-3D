# Data formats

## Inputs

| Input | Formats | Config key |
|---|---|---|
| Video | anything ffmpeg demuxes; NVDEC decodes H.264, HEVC, VP8/9, AV1, MPEG-2/4, VC-1, MJPEG, other codecs fall back to CPU decoding | `ingest.video` |
| Flight log (optional) | CSV (auto-detected columns), DJI `.srt`, `.gpx`, JSON list | `ingest.telemetry` |

Telemetry is interpolated to each keyframe's timestamp; georeferencing needs at
least `geo.min_correspondences` GPS-tagged keyframes per SfM model.

## Intermediate artifacts (inside the run directory)

| Path | Format |
|---|---|
| `dataset/images/pass_NN/f_XXXXXX.jpg` | keyframes, source resolution (letterbox cropped), nvJPEG q95; `XXXXXX` is the source frame index |
| `dataset/sparse/N/{cameras,images,points3D}.bin` | COLMAP binary model (spirula-studio writes it); one per reconstructable pass or merged group of passes |
| `dataset/depths/pass_NN/f_XXXXXX.png` | 16-bit PNG, linear z-depth scaled so the 99.9th percentile is 65535, 0 = no data (spirula-studio's format) |
| `dataset/depth_raw/pass_NN/f_XXXXXX.npz` | raw Marigold v2 log-depth prediction (`pred`, float16, 1024 px long side) |
| `keyframes/selection.json` | passes, keyframes, per-frame flow consistency and sharpness, selector config |
| `keyframes/keyframes.json` | one row per keyframe: name, frame index, time, pass, overlap, GPS if any |

## Outputs

| Path | Format |
|---|---|
| `splats/model_N/step-*.ckpt/splat.ply` | 3D Gaussian Splatting PLY (INRIA layout: position, normal, SH degree 3 `f_dc`/`f_rest`, logit opacity, log scale, wxyz rotation), float32 little-endian |
| `splats/model_N/scene_transform.json` | the similarity from the COLMAP frame to the frame the splats were trained in |
| `splats/model_N/mesh*.{ply,obj,glb}` | meshes extracted from the splats: vertex colour (PLY/GLB) and texture atlas (OBJ/GLB) |
| `splats/model_N/eval-{gt,render}-*.png`, `metrics.json` | held-out views (every 8th keyframe) and their PSNR / SSIM / L1 |
| `georef/model_N_sparse_enu.ply` | tie points in local East-North-Up metres; the WGS84 origin is in the PLY header comment and `georef/result.json` |
| `georef/camera_track.geojson` | GPS track of the keyframes (RFC 7946, WGS84) |
| `render/model_N_flythrough.mp4` | H.264 (NVENC) fly-through along the camera track |
| `metrics/metrics.json` | headline numbers of every stage, including per-stage GPU and host-memory use |
