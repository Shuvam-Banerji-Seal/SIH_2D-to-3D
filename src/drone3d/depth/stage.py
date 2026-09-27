"""The ``depth`` stage: Marigold v2 on every registered keyframe, aligned to SfM.

For each image of each SfM model it writes

* ``depths/<image stem>.png`` -- aligned linear depth in spirula's format, with
  pixels far beyond the reconstructed range (sky) left at 0;
* ``depth_raw/<image stem>.npz`` -- the raw prediction (float16) for re-use;

and returns per-image and per-model agreement with the SfM points (AbsRel,
delta1), the measurement of how far the monocular prior can be trusted.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["DepthStageResult", "aggregate", "configure_bitsandbytes", "recalibrate", "run_depth", "supervision_mask"]

log = get_logger(__name__)


def aggregate(per_image: dict[str, dict]) -> dict[str, dict]:
    """Per-model summary of the per-image records, including models where every alignment failed."""
    by_model: dict[str, list[dict]] = {}
    for record in per_image.values():
        by_model.setdefault(record["model"], []).append(record)
    out = {}
    for model, records in sorted(by_model.items()):
        ok = [r for r in records if "abs_rel" in r]

        def med(key: str, rows: list[dict] = ok) -> float | None:
            return round(float(np.median([r[key] for r in rows])), 4) if rows else None

        summary = {
            "images": len(records),
            "aligned": len(ok),
            "abs_rel_median": med("abs_rel_median"),
            "delta1_median": med("delta1"),
            "valid_fraction_mean": round(float(np.mean([r["valid_fraction"] for r in ok])), 4)
            if ok
            else None,
        }
        for kind in ("affine", "monotone"):
            for metric in ("abs_rel_median", "delta1"):
                vals = [r["cv"][kind][metric] for r in ok if "cv" in r]
                summary[f"cv_{kind}_{metric}"] = round(float(np.median(vals)), 4) if vals else None
        out[model] = summary
    return out


def supervision_mask(
    pred: np.ndarray, depth: np.ndarray, p_at: np.ndarray, z: np.ndarray, far_factor: float, margin: float = 0.05
) -> np.ndarray:
    """Pixels whose depth the SfM calibration actually constrains.

    The calibration is fitted where tie points exist; outside the range of
    predictions they cover (sky, featureless far field) its output is pure
    extrapolation. Those pixels, and anything beyond ``far_factor`` x the
    99th-percentile tie-point depth, get no supervision: an invented depth
    for the sky would ask the trainer to put splats there.
    """
    lo, hi = np.quantile(p_at, [0.005, 0.995])
    pad = margin * max(hi - lo, 1e-6)
    return (pred >= lo - pad) & (pred <= hi + pad) & (depth < far_factor * float(np.quantile(z, 0.99)))


def configure_bitsandbytes() -> None:
    """Drop a ``BNB_CUDA_VERSION`` that disagrees with torch's CUDA build.

    A stale value (e.g. ``126`` under a CUDA 13.2 torch) makes bitsandbytes
    load kernels for the wrong runtime; unset, it picks the matching library.
    Only this process's environment is touched.
    """
    import torch

    forced = os.environ.get("BNB_CUDA_VERSION")
    if forced and torch.version.cuda and forced != torch.version.cuda.replace(".", ""):
        log.warning("ignoring BNB_CUDA_VERSION=%s (torch is CUDA %s)", forced, torch.version.cuda)
        os.environ.pop("BNB_CUDA_VERSION")


@dataclass
class DepthStageResult:
    images: int
    seconds: float
    seconds_inference: float
    per_model: dict[str, dict] = field(default_factory=dict)
    per_image: dict[str, dict] = field(default_factory=dict)
    processing_size: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "images": self.images,
            "seconds": round(self.seconds, 2),
            "seconds_inference": round(self.seconds_inference, 2),
            "images_per_second": round(self.images / self.seconds_inference, 3)
            if self.seconds_inference
            else None,
            "processing_size": self.processing_size,
            "per_model": self.per_model,
            "per_image": self.per_image,
        }


def run_depth(
    dataset: Path,
    model_dirs: list[Path],
    *,
    long_side: int = 1024,
    batch: int = 4,
    quantization: str = "4bit",
    far_factor: float = 3.0,
    out_long_side: int | None = 1920,
    calibration: str = "monotone",
    device: str = "cuda",
) -> DepthStageResult:
    """Predict, align and write depth for every image registered in ``model_dirs``.

    Args:
        far_factor: pixels predicted beyond ``far_factor`` x the image's 99th
            percentile SfM depth are treated as sky / out of range (written 0).
        out_long_side: long side of the written depth maps (spirula resizes
            bilinearly to the training size anyway).
        calibration: ``monotone`` (isotonic, default) or ``affine`` mapping
            from the prediction to log depth.
    """
    configure_bitsandbytes()
    import cv2
    import torch
    import torch.nn.functional as F

    from drone3d.depth.align import (
        MonotoneMap,
        cross_validate,
        fit_log_affine,
        sample_at,
        sfm_depth_observations,
        write_depth_png,
    )
    from drone3d.depth.marigold import MarigoldDepth, processing_size

    started = time.perf_counter()
    images_dir = dataset / "images"
    depth_dir = dataset / "depths"
    raw_dir = dataset / "depth_raw"
    jobs: list[tuple[str, object]] = []
    for model_dir in model_dirs:
        for obs in sfm_depth_observations(model_dir).values():
            jobs.append((str(model_dir), obs))
    if not jobs:
        return DepthStageResult(0, 0.0, 0.0)
    net = MarigoldDepth(quantization=quantization, device=device)
    first = jobs[0][1]
    size = processing_size(first.width, first.height, long_side)
    result = DepthStageResult(
        images=len(jobs), seconds=0.0, seconds_inference=0.0, processing_size=size
    )
    infer_time = 0.0
    for s in range(0, len(jobs), batch):
        chunk = jobs[s : s + batch]
        rgbs = []
        for _, obs in chunk:
            bgr = cv2.imread(str(images_dir / obs.name), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(images_dir / obs.name)
            rgbs.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        shapes = {tuple(r.shape) for r in rgbs}
        t0 = time.perf_counter()
        if len(shapes) == 1:
            preds = net.predict(torch.stack(rgbs).to(device), size)
        else:  # mixed resolutions: one at a time
            preds = torch.cat([net.predict(r[None].to(device), size) for r in rgbs])
        torch.cuda.synchronize()
        infer_time += time.perf_counter() - t0
        for (model_dir, obs), pred in zip(chunk, preds, strict=True):
            stem = Path(obs.name).with_suffix("")
            p_at = sample_at(pred, obs.uv, obs.width, obs.height)
            fit = fit_log_affine(p_at, obs.z)
            raw_path = raw_dir / f"{stem}.npz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_path, pred=pred[0].cpu().numpy().astype(np.float16))
            record = {"model": model_dir, "sfm_points": int(len(obs.z))}
            if fit is None or fit.a <= 0:
                record["status"] = "alignment-failed"
                result.per_image[obs.name] = record
                continue
            record.update(fit.to_dict())
            cv = cross_validate(p_at, obs.z)
            if cv is not None:
                record["cv"] = cv
            # Written depth uses the monotone calibration (better on held-out
            # tie points at landscape scale); the affine fit is kept for its
            # slope and as the fallback when the prediction has too little spread.
            try:
                calib = MonotoneMap(p_at, obs.z, slope=fit.a) if calibration == "monotone" else None
            except ValueError:
                calib = None
            record["calibration"] = "monotone" if calib is not None else "affine"
            if out_long_side and out_long_side < max(obs.width, obs.height):
                ow, oh = processing_size(obs.width, obs.height, out_long_side)
            else:
                ow, oh = obs.width, obs.height
            up = F.interpolate(pred[None], size=(oh, ow), mode="bilinear", align_corners=False)[
                0, 0
            ]
            log_depth = calib(up) if calib is not None else fit.a * up + fit.b
            depth = torch.exp(log_depth).cpu().numpy()
            valid = supervision_mask(up.cpu().numpy(), depth, p_at, obs.z, far_factor)
            record["valid_fraction"] = round(float(valid.mean()), 4)
            write_depth_png(depth_dir / f"{stem}.png", depth, valid)
            result.per_image[obs.name] = record
        log.info("depth %d/%d", min(s + batch, len(jobs)), len(jobs))
    result.per_model = aggregate(result.per_image)
    result.seconds = time.perf_counter() - started
    result.seconds_inference = infer_time
    net.close()
    return result


def recalibrate(
    dataset: Path,
    depth_result: dict,
    *,
    out_dir: str = "depths_v2",
    far_factor: float = 3.0,
    out_long_side: int | None = 1920,
) -> dict[str, float]:
    """Rewrite depth maps from the stored predictions (no network), current rules.

    Uses ``depth_raw/`` and the SfM models recorded per image, the monotone
    calibration and :func:`supervision_mask`. Returns per-image valid fractions.
    """
    import torch
    import torch.nn.functional as F

    from drone3d.depth.align import (
        MonotoneMap,
        fit_log_affine,
        sample_at,
        sfm_depth_observations,
        write_depth_png,
    )
    from drone3d.depth.marigold import processing_size

    fractions: dict[str, float] = {}
    obs_by_model: dict[str, dict] = {}
    for name, rec in depth_result["per_image"].items():
        if "abs_rel" not in rec:
            continue
        model = rec["model"]
        if model not in obs_by_model:
            obs_by_model[model] = sfm_depth_observations(model)
        obs = obs_by_model[model][name]
        stem = Path(name).with_suffix("")
        pred = torch.from_numpy(np.load(dataset / "depth_raw" / f"{stem}.npz")["pred"].astype(np.float32))[None]
        p_at = sample_at(pred, obs.uv, obs.width, obs.height)
        fit = fit_log_affine(p_at, obs.z)
        if fit is None or fit.a <= 0:
            continue
        calib = MonotoneMap(p_at, obs.z, slope=fit.a)
        if out_long_side and out_long_side < max(obs.width, obs.height):
            ow, oh = processing_size(obs.width, obs.height, out_long_side)
        else:
            ow, oh = obs.width, obs.height
        up = F.interpolate(pred[None], size=(oh, ow), mode="bilinear", align_corners=False)[0, 0]
        depth = torch.exp(calib(up)).numpy()
        valid = supervision_mask(up.numpy(), depth, p_at, obs.z, far_factor)
        write_depth_png(dataset / out_dir / f"{stem}.png", depth, valid)
        fractions[name] = float(valid.mean())
    return fractions
