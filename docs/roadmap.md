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
