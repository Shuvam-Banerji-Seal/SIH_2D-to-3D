# Promo film storyboard — "One Pass. One GPU. One World in 3D."

Explainer, ~80 s, 1920×1080, 30 fps. Built with the javascript-animation skill
(every frame of the animation drawn in canvas code, synthesized soundtrack) and
then composited with **real pipeline output** in the "screen" slots below.

## Look and sound brief

| Choice | Decision | Why |
|---|---|---|
| Style | Cinematic flat + blueprint (technique 7) | An engineering pipeline reads as a technical drawing; the one style tested across a full film |
| Palette | night navy `#0b1d33` (ground), blueprint cyan `#5ad1ff` (lines), paper white `#f2f4f7` (text), saffron `#ff9933` (lead accent: the drone, the "3D" verdict), leaf green `#3fb950` (OK marks) | Dark ground makes real footage in the slots pop; saffron nods to SIH / India without a flag |
| Line | Thin cyan construction lines, low wobble (amp 0.8), boil every 4 frames | Precise, still alive |
| Type | Condensed sans (system `Inter, Helvetica, Arial`), labels as blueprint call-outs | Legible over footage |
| Framing | One continuous blueprint "world" that the camera travels along (left → right = pipeline order) | Screen direction carries the chain: the pipeline is literally a journey |
| Pace | 3–7 s shots | Explainer pace |
| Sound | `groove`, kit `electro`, harmony `bright`, 96 bpm | Confident tech-launch energy, steady grid so cuts land |

## Beat grid

96 bpm → 0.625 s per beat, a bar is 2.5 s. 32 bars = 80 s.

| Section | Bars | Seconds | Shape |
|---|---|---|---|
| Hook | 1–4 | 0–10 | the problem |
| Steps | 5–26 | 10–65 | one pipeline step per shot, a motif each |
| Break | 27 | 65–67.5 | silence before the payoff |
| Payoff | 28–32 | 67.5–80 | the 3D world, the recap |

## Shot list (times in seconds; ⧉ = composited real output)

| # | from–to | framing | camera | action | ⧉ slot | sound |
|---|---|---|---|---|---|---|
| 1 | 0–5 | wide | slow push | A drone (saffron) crosses a blueprint landscape once, left to right, trailing a dashed flight line. Caption: "One pass. One chance." | — | whoosh on the drone crossing |
| 2 | 5–10 | medium | drift | The flight line becomes a film strip; three frames lift off it. Caption: "Can one video become an accurate 3D model?" | ⧉ S1: source footage (Jal Mahal, 5 s) | film-strip ticks |
| 3 | 10–17.5 | close | push in | The strip runs through a GPU chip glyph; frames come out as NV12 → RGB tiles. Counter: frames/s. "One NVDEC decode feeds the analysis and the keyframes." | — | clicks on each tile |
| 4 | 17.5–25 | medium | pan right | Arrows (optical flow) appear between consecutive frames; a cut makes the arrows snap to red → the strip splits into passes. Caption: "Find the passes". | ⧉ S3: keyframe_timeline.png (top panel) | snap on the cut |
| 5 | 25–32.5 | close | slow push | A grid of dots rides the flow from keyframe K; the overlap gauge drains 1.0 → 0.75; at the band the next keyframe lights up saffron. Caption: "Similar enough to match. Different enough to see depth." | ⧉ S4: overlap panel | tick per keyframe |
| 6 | 32.5–40 | medium | pan right | Two views; a homography sheet peels off the flat ground but the palace pops out of it (parallax). Gauge: "parallax SNR 9.8 → 3D ✓". Contrast inset: spinning-camera control, "SNR 1.2 → not 3D". | — | rising chord on ✓ |
| 7 | 40–47.5 | wide | pull out | Keyframes become camera frusta above the model assembling point by point. "Camera poses from the flow itself" — no feature matching; registered count and agreement with feature SfM. | — | sparkle per burst |
| 8 | 47.5–55 | medium | drift | "Depth from parallax, completed by a prior": real 2×2 tiles — keyframe, triangulated depth, completed depth, the textured mesh rendered from the same pose. | ⧉ S6: depth tiles (promo/assets.py) | whoosh |
| 9 | 55–62.5 | close | push in | Soft blobs settle into one surface (TSDF fusion). Counter: triangles; "N % of what the camera saw". | ⧉ S7: textured-mesh fly-through inset | swell |
| 10 | 62.5–65 | wide | hold | Map pin + GPS track snaps the model level onto a grid; accuracy near the track, the six formats, processing time for the video. | — | click |
| 11 | 65–67.5 | black | — | break: silence | — | silence |
| 12 | 67.5–77.5 | full frame | — | The finished textured model, flown along the drone's own path. | ⧉ S8: ray-cast fly-through (10 s) | payoff groove |
| 13 | 77.5–80 | wide | hold | Recap line: "Video → keyframes → poses → depth → textured mesh → map." + "SIH 2026 · PS 26158" | — | resolve |

Slot geometry (world-to-screen rectangles, in 1920×1080 pixels) is exported by
the page as `window.SLOTS`, so the compositor places the footage exactly where the
animation draws the screen bezel.

Every fact on screen comes from a run directory in `outputs/` (see the README in
this folder).
