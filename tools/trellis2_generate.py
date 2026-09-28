"""TRELLIS.2 image-to-3D on one keyframe -> a textured GLB (a generated asset, not a measurement).

    third_party/TRELLIS.2/.venv/bin/python tools/trellis2_generate.py OUT.glb IMAGE [--res 512|1024] [--seed 0]

Runs in TRELLIS.2's own environment (tools/setup_trellis2.sh). Weights come from $HF_HOME
(/store/huggingface): microsoft/TRELLIS.2-4B, the TRELLIS-image-large decoder, DINOv3 ViT-L and RMBG-2.0
(the last two gated; the token is read from the repository's .env). The background is removed by RMBG-2.0
unless the image already has an alpha channel.

One image only: averaging the flow prediction over five Colosseum keyframes from different sides (TRELLIS v1's
"multidiffusion" multi-image mode) stacked several rings on each other -- TRELLIS.2 generates in a frame tied
to the input view, so views from different sides disagree about where the walls are.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").is_file() else []:
    key, _, value = line.partition("=")
    if key and value:
        os.environ.setdefault(key.strip(), value.strip())
os.environ.setdefault("HF_HOME", "/store/huggingface")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
sys.path.insert(0, str(ROOT / "third_party" / "TRELLIS.2"))


def subject(pipe, image):  # type: ignore[no-untyped-def]
    """RGBA of the building alone: RMBG-2.0's foreground, only its largest part.

    On an aerial keyframe RMBG keeps the Colosseum and, joined by a strip of street, a block of houses
    behind it; TRELLIS.2 then generated both. An opening at 3.5 % of the diagonal cuts the strip.
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 255:
        return image
    rgb = image.convert("RGB")
    scale = min(1, 1024 / max(rgb.size))
    rgb = rgb.resize((int(rgb.width * scale), int(rgb.height * scale)), Image.Resampling.LANCZOS)
    pipe.rembg_model.to(pipe.device)
    alpha = np.asarray(pipe.rembg_model(rgb.copy()))[..., 3].copy()  # it puts the alpha into its input
    # an opening: parts joined to the largest only through a neck narrower than 7 % of the diagonal are cut
    fg = alpha > 127
    r = max(1, int(0.035 * np.hypot(*fg.shape)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = yy**2 + xx**2 <= r * r
    labels, n = ndimage.label(ndimage.binary_erosion(fg, disk))
    if n:
        largest = labels == 1 + int(np.argmax(ndimage.sum(labels > 0, labels, index=range(1, n + 1))))
        alpha[~(ndimage.binary_dilation(largest, disk) & fg)] = 0
    return Image.fromarray(np.dstack([np.asarray(rgb), alpha]), "RGBA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("image")
    ap.add_argument("--res", default="1024", choices=["512", "1024"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import o_voxel
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from trellis2.modules import image_feature_extractor as fe
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:  # transformers >= 5 keeps the blocks in .model.layer
        model = self.model
        image = image.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden = model.embeddings(image, bool_masked_pos=None)
        rope = model.rope_embeddings(image)
        for block in getattr(model, "layer", None) or model.model.layer:
            hidden = block(hidden, position_embeddings=rope)
        return F.layer_norm(hidden, hidden.shape[-1:])

    fe.DinoV3FeatureExtractor.extract_features = extract_features

    t0 = time.perf_counter()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    kind = "512" if args.res == "512" else "1024_cascade"
    cut = subject(pipe, Image.open(args.image))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cut.save(Path(args.out).with_suffix(".input.png"))  # what was generated from, beside the GLB
    mesh = pipe.run(cut, seed=args.seed, pipeline_type=kind)[0]
    mesh.simplify(16777216)
    t_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                     attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                     decimation_target=500000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0,
                                     verbose=False)  # fmt: skip
    glb.export(args.out)
    print(f"{args.out}: load {t_load:.1f} s, generate {t_gen:.1f} s, export {time.perf_counter() - t0:.1f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
