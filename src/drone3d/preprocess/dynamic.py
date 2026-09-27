"""Dynamic-object masking so vehicles, people and animals do not pollute the model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["DynamicMasker", "YoloMasker", "denoise_mask"]

log = get_logger(__name__)


def denoise_mask(mask: np.ndarray, *, min_area: float = 400.0, kernel_size: int = 3) -> np.ndarray:
    """Clean a binary mask and drop connected components smaller than ``min_area``."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    output = np.zeros_like(cleaned)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 255
    return output


class DynamicMasker:
    """Background-subtraction masker tuned for a moving UAV camera.

    A per-pixel mixture-of-Gaussians model is used as a fast, dependency-light
    approximation; the optional :class:`YoloMasker` provides semantic masks
    when the ``ai`` extra is installed.

    Caveat (F24): background subtraction assumes a *static* camera. On a
    translating UAV the whole scene moves, so MOG2 flags static terrain as
    foreground — measured at 67–74 % of the frame on the bundled clip. Such a
    mask is not "dynamic objects"; it is camera motion, and discarding it would
    delete most of the scene from SfM. ``max_dynamic_fraction`` therefore
    treats an implausibly large mask as a failed detection and returns an empty
    one, keeping the frame intact.
    """

    def __init__(
        self,
        *,
        history: int = 120,
        var_threshold: float = 24.0,
        min_area: float = 400.0,
        dilate_iterations: int = 2,
        max_dynamic_fraction: float = 0.5,
    ) -> None:
        self.min_area = min_area
        self.dilate_iterations = dilate_iterations
        self.max_dynamic_fraction = max_dynamic_fraction
        self._model = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a 0/255 mask where 255 marks likely dynamic pixels."""
        foreground = self._model.apply(frame, learningRate=0.02)
        foreground[foreground == 127] = 0  # drop shadows
        mask = denoise_mask(foreground, min_area=self.min_area)
        if self.dilate_iterations:
            mask = cv2.dilate(mask, self._kernel, iterations=self.dilate_iterations)
        if self.max_dynamic_fraction > 0:
            fraction = float(np.count_nonzero(mask)) / float(mask.size)
            if fraction > self.max_dynamic_fraction:
                log.debug(
                    "dynamic mask covers %.0f%% of the frame -- treating as camera "
                    "motion, not movers; emitting empty mask",
                    fraction * 100,
                )
                return np.zeros_like(mask)
        return mask

    def moving_ratio(self, frame: np.ndarray) -> float:
        """Fraction of the frame covered by the dynamic mask."""
        mask = self.mask(frame)
        return float(np.count_nonzero(mask)) / float(mask.size)


class YoloMasker:
    """Semantic dynamic-object masker backed by Ultralytics YOLO (optional)."""

    DEFAULT_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 8)  # person..truck, bus, car

    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        *,
        classes: tuple[int, ...] | None = None,
        confidence: float = 0.25,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("YoloMasker requires the 'ai' extra: uv sync --extra ai") from exc
        self._model = YOLO(str(model))
        self.classes = classes if classes is not None else self.DEFAULT_CLASSES
        self.confidence = confidence

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a 0/255 mask covering detected dynamic objects."""
        results = self._model.predict(frame, conf=self.confidence, verbose=False)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for box, class_id in zip(
                boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), strict=False
            ):
                if int(class_id) not in self.classes:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box)
                mask[max(0, y1) : y2, max(0, x1) : x2] = 255
        return mask
