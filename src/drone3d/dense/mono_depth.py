"""Monocular depth estimation backend for low-overlap or occluded regions.

This backend does not produce a metric cloud on its own; it writes relative
depth maps that can (a) drive confidence-weighted fusion in a later MVS pass or
(b) support a scale-calibrated monocular surface prior. Metric output still
requires the SfM model and GPS priors handled by the georeferencing stage.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone3d.config import DenseConfig
from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.logging_utils import get_logger
from drone3d.types import DenseResult, SfMResult

__all__ = ["MonoDepthBackend", "MonoDepthEstimator"]

log = get_logger(__name__)


class MonoDepthEstimator:
    """Wraps a Hugging Face depth-estimation pipeline (Depth Anything V2 by default)."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailable(
                "monocular depth needs the 'ai' extra: uv sync --extra ai"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self._pipeline = pipeline(task="depth-estimation", model=model_id, device=device)

    def depth(self, image: np.ndarray) -> np.ndarray:
        """Return a float32 relative depth map normalised to ``[0, 1]``.

        Polarity: **1 = nearest, 0 = farthest.** Depth Anything V2 predicts
        inverse depth (disparity), and the min-max normalisation below preserves
        that polarity, so larger values are *closer*. Verified empirically on
        2026-09-26 with ``Depth-Anything-V2-Small-hf``: a street photo's near
        bottom half normalised to 0.63 against 0.31 for the far top half.
        """
        from PIL import Image  # provided by transformers

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._pipeline(Image.fromarray(rgb))
        depth = np.asarray(result["depth"], dtype=np.float32)
        minimum, maximum = float(depth.min()), float(depth.max())
        if maximum - minimum < 1e-6:
            return np.zeros_like(depth)
        return (depth - minimum) / (maximum - minimum)


class MonoDepthBackend:
    """Writes normalised per-frame depth maps for downstream fusion."""

    name = "mono"

    def __init__(self, device: str = "auto") -> None:
        self.device = device

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def reconstruct(
        self,
        images_dir: Path,
        sfm: SfMResult,
        output_dir: Path,
        config: DenseConfig,
    ) -> DenseResult:
        if not self.is_available():
            raise BackendUnavailable("monocular depth needs the 'ai' extra: uv sync --extra ai")
        images = sorted(
            path
            for path in Path(images_dir).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not images:
            raise ReconstructionError(f"no images found in {images_dir}")

        depth_dir = Path(output_dir) / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        estimator = MonoDepthEstimator(config.mono_depth_model, device=self.device)

        written = 0
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                log.warning("skipping unreadable image: %s", image_path)
                continue
            depth = estimator.depth(image)
            np.save(depth_dir / f"{image_path.stem}.npy", depth)
            written += 1
        log.info("wrote %d relative depth maps to %s", written, depth_dir)
        return DenseResult(
            backend=self.name,
            depth_dir=depth_dir,
            metadata={
                "model": config.mono_depth_model,
                "note": "relative depth; scale must come from SfM/GPS priors",
                "num_depth_maps": written,
            },
        )
