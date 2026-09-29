# Explainer storyboard — "How it works"

~152 s, 1920×1080, 30 fps. Built with the javascript-animation skill (every frame of the animation drawn
in canvas code, a synthesized score) and composited with **real output of our own runs** in the screens.

    uv run python promo/explainer_assets.py      # the footage (promo/build/explainer/*.mp4)
    uv run python promo/explainer_build.py       # facts -> page -> render -> composite: build/explainer/explainer.mp4

## Look and sound brief (a different piece from the promo, so a different look and sound)

| Choice | Decision | Why |
|---|---|---|
| Style | Technical poster on charcoal, a dot grid, clean lines (no wobble) | An explainer of an engineering pipeline: precise, calm |
| Palette | charcoal `#121417`, teal `#2ec4a8` = **measured**, violet `#a98bff` = **generated**, amber `#f5b700` = numbers | The colour code carries the film's one honesty rule: what was measured and what was generated |
| Type | Sans for statements, mono for data labels | Legible over footage, data reads as data |
| Framing | Left: step number, headline, three bullets; right: a screen of real output | Every claim sits next to its evidence |
| Pace | 8–16 s per step | Explainer pace: one idea per shot |
| Sound | `groove`, kit `keys` (brushes, electric piano, strings), harmony `dreamy`, 90 bpm | Warmer and calmer than the promo's electro; a steady beat so cuts land |

## Shots (bars of 2.67 s)

| # | bars | step | real footage in the screen (all ours) |
|---|---|---|---|
| 1 | 0–4 | title | the highrise orbit, the source video |
| 2 | 4–7 | the brief (time, accuracy, completeness, formats) | — |
| 3 | 7–11 | 1 · decode & keyframes | the highrise run's keyframe timeline (flow, overlap, keyframes) |
| 4 | 11–15 | 2 · poses from optical flow | — (a drawn orbit of cameras; 41/41 placed) |
| 5 | 15–19 | 3 · many shots, one model | Colosseum: 11 shots meeting in one model |
| 6 | 19–23 | 4 · depth | Colosseum: keyframe, triangulated, completed depth, mesh |
| 7 | 23–27 | 5 · fused, then clean | highrise: the fused mesh wiped into the clean model (shaded) |
| 8 | 27–31 | 6 · texture | Colosseum: the textured clean model flown along the drone's path |
| 9 | 31–37 | 7 · completing the unseen | the 110° arc diagram, then the highrise's complete model circled 360° |
| 10 | 37–41 | 8 · 360° splats | the highrise's 360° Gaussian splats circled |
| 11 | 41–44 | 9 · deliverables & console | the console's explorer with the complete-model layer |
| 12 | 44–47 | speed | drawn: the 15 sample videos, time vs budget |
| 13 | 47–48 | break | — |
| 14 | 48–55 | showcase | Notre-Dame, the Reichstag block, rice fields (flown along their flights), the highrise complete |
| 15 | 55–57 | recap | — |

Examples were picked from stills along each flight for full terrain and no open backs: a flythrough
follows the drone's own path, so it shows what was captured; the only turntable is of the complete model.
