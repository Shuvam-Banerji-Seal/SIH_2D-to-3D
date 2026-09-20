"""Feature-based global motion estimation and video stabilization."""

from __future__ import annotations

import cv2
import numpy as np

from drone3d.logging_utils import get_logger

__all__ = ["Stabilizer", "estimate_affine", "smooth_trajectory", "stabilize_frames"]

log = get_logger(__name__)


def estimate_affine(previous_gray: np.ndarray, gray: np.ndarray) -> np.ndarray | None:
    """Estimate the partial-affine transform mapping ``gray`` onto ``previous_gray``."""
    detector = cv2.ORB_create(nfeatures=800)
    keypoints_a, descriptors_a = detector.detectAndCompute(previous_gray, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray, None)
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_b, descriptors_a), key=lambda m: m.distance)[:200]
    if len(matches) < 8:
        return None

    source = np.float32([keypoints_b[m.queryIdx].pt for m in matches])
    target = np.float32([keypoints_a[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 6:
        return None
    return matrix


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compose two affine transforms into a 3x3 matrix product."""
    outer_3x3 = np.vstack([outer, [0.0, 0.0, 1.0]])
    inner_3x3 = np.vstack([inner, [0.0, 0.0, 1.0]])
    return (outer_3x3 @ inner_3x3)[:2]


def smooth_trajectory(transforms: list[np.ndarray], window: int = 15) -> list[np.ndarray]:
    """Low-pass filter cumulative transforms by averaging their parameters."""
    if not transforms:
        return []
    window = max(1, window)
    kernel = np.ones(window) / window
    parameters = np.array(
        [np.array([t[0, 2], t[1, 2], np.arctan2(t[1, 0], t[0, 0])]) for t in transforms]
    )
    padded = np.pad(parameters, ((window // 2, window // 2), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(parameters.shape[1])]
    )[: len(transforms)]

    result: list[np.ndarray] = []
    for transform, params in zip(transforms, smoothed, strict=True):
        scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        angle = float(params[2])
        result.append(
            np.array(
                [
                    [scale * np.cos(angle), -scale * np.sin(angle), params[0]],
                    [scale * np.sin(angle), scale * np.cos(angle), params[1]],
                ],
                dtype=np.float64,
            )
        )
    return result


class Stabilizer:
    """Causal stabilizer that keeps a smoothed virtual camera trajectory."""

    def __init__(self, *, smoothing: int = 15, border_mode: int = cv2.BORDER_REFLECT) -> None:
        self.smoothing = smoothing
        self.border_mode = border_mode
        self._cumulative = np.eye(3, dtype=np.float64)[:2]
        self._history: list[np.ndarray] = []
        self._previous_gray: np.ndarray | None = None

    def _cumulative_list(self) -> list[np.ndarray]:
        return list(self._history)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize a single BGR frame in a streaming fashion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is not None:
            relative = estimate_affine(self._previous_gray, gray)
            if relative is not None:
                self._cumulative = _compose(self._cumulative, relative)
        self._previous_gray = gray

        self._history.append(self._cumulative.copy())
        smoothed = smooth_trajectory(self._history, self.smoothing)[-1]
        correction = _compose(np.linalg.inv(np.vstack([self._cumulative, [0, 0, 1]]))[:2], smoothed)
        height, width = frame.shape[:2]
        return cv2.warpAffine(
            frame,
            correction,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
        )


def stabilize_frames(
    frames: list[np.ndarray],
    *,
    smoothing: int = 15,
) -> list[np.ndarray]:
    """Stabilize a batch of BGR frames using a smoothed cumulative trajectory."""
    if len(frames) < 2:
        return list(frames)

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    cumulative: list[np.ndarray] = [np.eye(3, dtype=np.float64)[:2]]
    for index in range(1, len(frames)):
        relative = estimate_affine(grays[index - 1], grays[index])
        if relative is None:
            relative = np.eye(3, dtype=np.float64)[:2]
        cumulative.append(_compose(cumulative[-1], relative))

    smoothed = smooth_trajectory(cumulative, smoothing)
    output: list[np.ndarray] = []
    for frame, accumulated, filtered in zip(frames, cumulative, smoothed, strict=True):
        accumulated_3x3 = np.vstack([accumulated, [0.0, 0.0, 1.0]])
        correction = _compose(np.linalg.inv(accumulated_3x3)[:2], filtered)
        height, width = frame.shape[:2]
        output.append(
            cv2.warpAffine(
                frame,
                correction,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    log.info("stabilized %d frames (smoothing=%d)", len(output), smoothing)
    return output
