// Method: what the system does to a video, and how each number in the console is measured.
export function viewAbout(main) {
  const stages = [
    ['Decode', 'One NVDEC pass yields analysis frames and full-size keyframe candidates (nvJPEG) together; the analysis rate adapts to how fast the camera moves.', 'NVDEC · nvJPEG'],
    ['Keyframes', 'RAFT optical flow measures overlap; keyframes are placed in an overlap band and every pass gets a parallax verdict — pure rotations are refused, not faked.', 'RAFT large · CUDA graphs'],
    ['Poses', 'Sub-pixel flow tracks, checked against direct flow, feed COLMAP’s global mapper (GLOMAP) per pass, while the GPU tracks the next pass.', 'flow tracks · GLOMAP'],
    ['Depth', 'Every keyframe is triangulated against neighbours 2–12 keyframes away; Depth Anything V2 fills what flow cannot see, calibrated per image. Sky stays empty.', 'flow triangulation · DA-V2 L'],
    ['Surface', 'Depth maps fuse in a GPU TSDF into a watertight mesh and a dense cloud; each keyframe’s photo is baked onto the mesh.', 'Open3D CUDA TSDF · GPU texture'],
    ['Splats', 'Optional: 3D Gaussian Splatting starts from the dense cloud and trains in seconds on the same poses, for photoreal fly-throughs.', 'spirula-studio · 3DGS'],
    ['Georeference', 'With a flight log, a similarity to the GPS track puts every model in ENU metres and UTM (EPSG) — LAS and GeoTIFF carry the code.', 'SRT · CSV · GPX · JSON'],
    ['Deliver', 'OBJ, PLY, LAS, GeoTIFF DSM + orthophoto, GLB, FBX, web splats and this viewer, per model.', '< 15 min per 10-min video'],
  ];
  main.innerHTML = `
    <div class="page-head rise"><h1>Method<small>Single-pass drone video to a complete, measurable 3D model on one GPU — SIH 2026 problem statement 26158 (NTRO).</small></h1></div>
    <section class="panel rise" style="--i:1"><h2>The pipeline</h2><div class="pipe">${stages.map(([t, d, e]) => `<div class="p"><b>${t}</b><span>${d}</span><em>${e}</em></div>`).join('')}</div></section>
    <div class="grid g-12" style="margin-top:16px">
      <section class="panel s6 rise prose" style="--i:2"><h2>Problem statement targets</h2>
        <p><b>Speed</b> — under 15 minutes of processing for a 10-minute video, i.e. 1.5× the video's length. Every run shows its time against that budget; the fast profile is tuned to it (Jal Mahal, 55 s of 4K: 76 s against 82 s on an idle A100).</p>
        <p><b>Spatial accuracy</b> — ≤ 1 m. With a flight log the models are metric and georeferenced; the metrics stage measures accuracy and completeness against a ground-truth cloud (LAS or PLY) within 1 m when one is given.</p>
        <p><b>Completeness</b> — the share of every registered keyframe's non-sky pixels whose ray hits the reconstructed mesh, computed exactly by ray casting.</p>
        <p><b>Formats</b> — OBJ, PLY, LAS, GeoTIFF, glTF/GLB and FBX, plus this viewer.</p></section>
      <section class="panel s6 rise prose" style="--i:3"><h2>Engine, live footage, safety</h2>
        <p>The <b>engine</b> is a resident GPU process: RAFT and Depth Anything stay loaded between runs and RAFT's CUDA graphs are captured once. It loads a model only if the GPU has its footprint plus a reserve free <i>now</i> — other users' work is never crowded out — and fits memory-hungry settings to the free memory before each run. A sticky CUDA fault makes it save its queue and restart.</p>
        <p><b>Live</b> footage (RTSP, RTMP, SRT, UDP, HLS, a camera, or a recorded flight replayed at its frame rate) is cut into segments at keyframes without re-encoding; each closed segment becomes a run, ahead of other work, so a model exists while the drone is still flying.</p>
        <p><b>Explorer</b>: <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to fly, <i>follow flight</i> to replay the drone's own path, click a keyframe to look through it with the photo over the model, measure distances and heights.</p></section>
    </div>`;
}
