# Roadmap

## Done (on branch `gs-pipeline`)

- GPU ingest (NVDEC, GPU colour conversion, nvJPEG), letterbox and fade handling.
- Pass segmentation, overlap-band keyframes, direct-flow parallax test with a
  pure-rotation negative control.
- spirula-studio GPU SfM, depth-supervised 3DGS, meshing; gsplat fly-throughs.
- Marigold v2 depth prior with per-image monotone calibration and held-out scoring.
- Ground-levelled georeferencing with leave-one-out error and scale uncertainty
  (validated in simulation).
- Per-stage GPU / host-memory telemetry; paper and promo film generated from run JSON.

## Next

0. **Fast profile within the official budget (< 15 min for a 10-minute video).**
   The 3DGS profile above takes hours for a 55 s clip and produces splats the
   problem statement does not ask for; the deliverable is a mesh / point cloud
   (OBJ, PLY, LAS, GeoTIFF, glTF/GLB, FBX) in a web or desktop viewer, and
   processing speed is 20 % of the score. Plan, each step timed on real footage:
   - one decode pass: analysis frames for flow and working-resolution keyframe
     candidates (nvJPEG) from the same NVDEC session, no second 4K decode;
   - correspondences from the flow already computed (direct keyframe-pair RAFT,
     tracks carried across keyframes by sampling flow at sub-pixel positions)
     instead of SIFT extraction and matching;
   - global SfM (pycolmap global mapper) on those tracks;
   - dense depth per keyframe from flow triangulation with the SfM poses, filled
     by a fast monocular prior calibrated to it; GPU TSDF fusion (Open3D) to a
     coloured mesh and a dense point cloud;
   - exports: PLY, OBJ, GLB, LAS (laspy), GeoTIFF DSM + orthophoto, FBX; a
     self-contained web viewer; per-stage time against the budget in the report.
1. **A real flight log.** Every public sample lacks GPS; georeferencing accuracy
   on real telemetry (DJI SRT) is the most important missing evaluation. The
   SRT parser also has known gaps for older DJI formats (`longtitude`,
   `GPS(lon,lat,alt)`).
2. **Dynamic objects.** Wire spirula-studio's `distraction_robustness` or SAM text
   masks ("person; car; boat") into the pipeline and measure ghosting.
3. **Far-field depth.** Marigold saturates on distant scenes; a metric depth
   model or a learned monotone calibration across images could replace per-image
   withholding.
4. **Throughput.** Overlap decode of the next clip with training of the current
   one; train small pass models with fewer steps.
5. **Compositional objects.** WorldSculpt-style per-object meshes from the splat
   scene given instance masks and boxes.
