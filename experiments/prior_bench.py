"""Which depth prior predicts the geometry a drone pass measures? Held-out error against flow-triangulated depth.

    third_party/priors/{da3,moge}/.venv/bin/python experiments/prior_bench.py [model ...]   (tools/setup_priors.sh; reads
                                                                       outputs/experiments/prior_ref.npz from
                                                                       experiments/prior_dump.py; writes paper/figures/prior_bench.json)

Each prior's per-keyframe prediction is calibrated to the triangulated depth on one part of the triangulated pixels
and scored on the rest, two ways: a robust log-affine fit of log depth, and the fill's own calibration (the
monotone map of drone3d.depth.align, which bends where an affine-invariant disparity is not affine in log depth):

- ``random``: half of the pixels calibrate, the other half score (how well the prior's shape matches);
- ``far``: the nearest 80 % calibrate, the farthest 20 % score (how it extrapolates, which is what the fill does).

Scores are median |z_pred - z| / z per keyframe, then the median over keyframes. Runs outside the drone3d venv (the
priors need an older torch), so it depends only on numpy, PIL and the priors' own packages.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # drone3d.depth.align: numpy (and optional torch) only


# ----------------------------------------------------------------------------- priors: image (PIL) -> log depth
def _da2():
    import torch
    from transformers import AutoModelForDepthEstimation

    net = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda().eval().half()
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.inference_mode()
    def run(img: Image.Image, size: tuple[int, int]) -> np.ndarray:  # disparity -> "log depth" up to an affine map
        w, h = size
        tw, th = max(14, round(w * 518 / max(w, h) / 14) * 14), max(14, round(h * 518 / max(w, h) / 14) * 14)
        x = torch.from_numpy(np.asarray(img.convert("RGB").resize((tw, th), Image.BICUBIC))).cuda().permute(2, 0, 1)[None].float() / 255
        disp = net(pixel_values=((x - mean) / std).half()).predicted_depth[:, None].float()
        disp = torch.nn.functional.interpolate(disp, size=(h, w), mode="bilinear")[0, 0].clamp_min(1e-6)
        return -torch.log(disp).cpu().numpy()

    return run


def _moge(repo: str):
    import torch
    from moge.model import import_model_class_by_version

    version = "v3" if "moge-3" in repo else "v2"
    net = import_model_class_by_version(version).from_pretrained(repo).cuda().eval()

    @torch.inference_mode()
    def run(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
        x = torch.from_numpy(np.asarray(img.convert("RGB").resize(size, Image.BICUBIC))).cuda().permute(2, 0, 1).float() / 255
        out = net.infer(x, use_fp16=True)
        d = out["depth"].float()
        return torch.log(torch.where(torch.isfinite(d) & (d > 0), d, torch.full_like(d, float("nan")))).cpu().numpy()

    return run


def _da3(repo: str):
    import torch
    from depth_anything_3.api import DepthAnything3

    net = DepthAnything3.from_pretrained(repo).to("cuda").eval()

    @torch.inference_mode()
    def run(img: Image.Image, size: tuple[int, int]) -> np.ndarray:  # one view at a time: the prior's monocular use
        pred = net.inference([np.asarray(img.convert("RGB"))])
        d = torch.from_numpy(np.asarray(pred.depth[0], dtype=np.float32))[None, None]
        d = torch.nn.functional.interpolate(d, size=(size[1], size[0]), mode="bilinear")[0, 0].clamp_min(1e-6)
        return torch.log(d).numpy()

    return run


PRIORS = {
    "depth-anything-v2-large": _da2,  # the fast profile's prior now
    "moge-3-vitl": lambda: _moge("Ruicheng/moge-3-vitl"),
    "moge-3-vitg": lambda: _moge("Ruicheng/moge-3-vitg"),
    "da3-large-1.1": lambda: _da3("depth-anything/DA3-LARGE-1.1"),
    "da3-metric-large": lambda: _da3("depth-anything/DA3METRIC-LARGE"),
    "da3-nested-giant-large-1.1": lambda: _da3("depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
}


# ----------------------------------------------------------------------------- calibration and scoring
def _fit(f: np.ndarray, lz: np.ndarray, iters: int = 5) -> tuple[float, float]:
    """Robust (IRLS, Huber) fit lz = a f + b."""
    w = np.ones_like(f)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([f * w, w], 1)
        a, b = np.linalg.lstsq(A, lz * w, rcond=None)[0]
        r = np.abs(lz - (a * f + b))
        k = 1.345 * max(float(np.median(r)) / 0.6745, 1e-6)
        w = np.sqrt(np.where(r <= k, 1.0, k / np.maximum(r, 1e-12)))
    return float(a), float(b)


def _score(f: np.ndarray, tri: np.ndarray, sky: np.ndarray, rng: np.random.Generator) -> dict:
    ok = (tri > 0) & ~sky & np.isfinite(f)
    ff, z = f[ok].astype(np.float64), tri[ok].astype(np.float64)
    from drone3d.depth.align import MonotoneMap

    out = {}
    cal = rng.random(len(z)) < 0.5  # random split
    near = z <= np.quantile(z, 0.8)  # extrapolation split: the nearest 80 % calibrate
    for split, fit in (("random", cal), ("far", near)):
        a, b = _fit(ff[fit], np.log(z[fit]))
        rel = lambda lz, fit=fit: float(np.median(np.abs(np.exp(lz) - z[~fit]) / z[~fit]))  # noqa: E731
        out[split] = rel(a * ff[~fit] + b)
        try:  # the fill's calibration, given the prior's robust slope as it is in drone3d.fastsfm.mono
            out[f"{split}_monotone"] = rel(MonotoneMap(ff[fit], z[fit], slope=a)(ff[~fit])) if a > 0 else float("nan")
        except ValueError:
            out[f"{split}_monotone"] = float("nan")
    return out


def main() -> None:
    names = sys.argv[1:] or list(PRIORS)
    ref = np.load(ROOT / "outputs" / "experiments" / "prior_ref.npz")
    index = json.loads((ROOT / "outputs" / "experiments" / "prior_ref.json").read_text())
    dst = ROOT / "paper" / "figures" / "prior_bench.json"
    rows = [r for r in (json.loads(dst.read_text()) if dst.is_file() else []) if r["prior"] not in names]
    for name in names:
        try:
            run = PRIORS[name]()
        except Exception as exc:  # a missing package or checkpoint: skip that prior, keep the others
            print(f"{name}: unavailable ({type(exc).__name__}: {str(exc)[:160]})", flush=True)
            continue
        rng = np.random.default_rng(0)
        per, secs = [], []
        for item in index:
            img = Image.open(item["image"])
            t0 = time.perf_counter()
            f = run(img, tuple(item["size"]))
            secs.append(time.perf_counter() - t0)
            s = _score(f, ref[f"tri/{item['key']}"], ref[f"sky/{item['key']}"], rng)
            per.append({"key": item["key"], **{k: round(v, 4) for k, v in s.items()}})
        row = {"prior": name, "keyframes": len(per), "random_median": round(float(np.median([p["random"] for p in per])), 4),
               "far_median": round(float(np.median([p["far"] for p in per])), 4),
               "random_monotone_median": round(float(np.nanmedian([p["random_monotone"] for p in per])), 4),
               "far_monotone_median": round(float(np.nanmedian([p["far_monotone"] for p in per])), 4),
               "seconds_per_image": round(float(np.median(secs[1:] or secs)), 3), "per_keyframe": per}  # fmt: skip
        rows.append(row)
        print({k: v for k, v in row.items() if k != "per_keyframe"}, flush=True)
        del run
    dst.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
