"""Does the static-overlay detector find logos and nothing else? Every sample video and upload.

    uv run python experiments/overlay_check.py OUT_DIR   (writes OUT_DIR/<video>.jpg and paper/figures/overlay_check.json)

Twelve frames sampled across each video (the keyframe stage's own probe times), the mask drawn in
magenta over their median; the share of the frame it covers and the boxes it found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (thread binding)
from drone3d.io.nvdec import analysis_size, probe_stream, sample_frames
from drone3d.io.overlay import static_overlay_mask

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from PIL import Image
    from scipy import ndimage

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for d in ("datasets", "uploads") for p in (ROOT / d).iterdir() if p.suffix in (".webm", ".mp4", ".mkv", ".mov"))
    rows = []
    for v in videos:
        info = probe_stream(v)
        size = analysis_size(info.width, info.height, 960)
        times = np.linspace(0.05, 0.95, 12) * info.duration_s
        frames = sample_frames(info, list(times), size)
        luma = [f[: size[0] * size[1]].reshape(size[1], size[0]) for f in frames if f is not None]  # NV12: luma first
        mask = static_overlay_mask(luma)
        med = np.median(np.stack(luma), 0).astype(np.uint8)
        rgb = np.stack([med] * 3, -1)
        boxes = []
        if mask is not None:
            rgb[mask] = (0.5 * rgb[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            lab, n = ndimage.label(mask)
            boxes = [[s[1].start, s[0].start, s[1].stop, s[0].stop] for s in ndimage.find_objects(lab)]
        Image.fromarray(rgb).resize((480, round(480 * size[1] / size[0]))).save(out / f"{v.stem[:40]}.jpg")
        row = {"video": v.name, "share": round(float(mask.mean()), 4) if mask is not None else 0.0, "boxes": boxes}
        rows.append(row)
        print(row, flush=True)
    (ROOT / "paper" / "figures" / "overlay_check.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
