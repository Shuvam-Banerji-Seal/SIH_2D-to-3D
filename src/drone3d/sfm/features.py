"""Feature detection, matching and frame-graph diagnostics (OpenCV only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from drone3d.logging_utils import get_logger

__all__ = [
    "build_frame_graph",
    "detect_features",
    "match_descriptors",
    "pair_overlap",
]

log = get_logger(__name__)


def detect_features(
    image: np.ndarray,
    *,
    method: str = "sift",
    max_features: int = 2000,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect keypoints and descriptors with SIFT, ORB or AKAZE."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectors = {
        "sift": cv2.SIFT_create(nfeatures=max_features),
        "orb": cv2.ORB_create(nfeatures=max_features),
        "akaze": cv2.AKAZE_create(),
    }
    detector = detectors.get(method.lower())
    if detector is None:
        raise ValueError(f"unsupported feature method '{method}' (sift, orb, akaze)")
    return detector.detectAndCompute(gray, None)


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    *,
    method: str = "sift",
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Ratio-test matching with L2 (SIFT/AKAZE) or Hamming (ORB) distance."""
    if descriptors_a is None or descriptors_b is None:
        return []
    norm = cv2.NORM_HAMMING if method.lower() == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    return [first for first, second in pairs if first.distance < ratio * second.distance]


def pair_overlap(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    method: str = "sift",
    max_features: int = 2000,
    ratio: float = 0.75,
    min_inliers: int = 12,
) -> dict[str, Any]:
    """Estimate the overlap between two frames from matched inlier keypoints.

    Returns:
        Mapping with ``matches``, ``inliers``, ``overlap`` (inlier fraction of
        the smaller keypoint set) and ``connected`` (RANSAC-verified overlap).
    """
    keypoints_a, descriptors_a = detect_features(image_a, method=method, max_features=max_features)
    keypoints_b, descriptors_b = detect_features(image_b, method=method, max_features=max_features)
    matches = match_descriptors(descriptors_a, descriptors_b, method=method, ratio=ratio)
    base = max(1, min(len(keypoints_a), len(keypoints_b)))

    inliers = 0
    if len(matches) >= 8:
        source = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
        target = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
        _, mask = cv2.findFundamentalMat(
            source, target, method=cv2.FM_RANSAC, ransacReprojThreshold=3.0
        )
        if mask is not None:
            inliers = int(mask.sum())
    return {
        "matches": len(matches),
        "inliers": inliers,
        "overlap": round(inliers / base, 4),
        "connected": inliers >= min_inliers,
    }


def build_frame_graph(
    frame_paths: list[Path],
    *,
    method: str = "sift",
    min_inliers: int = 12,
    max_pairs: int | None = 4000,
    neighbor_window: int = 4,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Build a frame adjacency graph from pairwise overlap estimates.

    Frames are compared with their temporal neighbours (single-pass video has
    strong local overlap) and, when the budget allows, with a strided sample of
    distant frames to catch revisited terrain.
    """
    images = []
    for path in frame_paths:
        image = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cannot read frame: {path}")
        images.append(image)

    pairs: list[tuple[int, int]] = []
    count = len(images)
    for i in range(count):
        for offset in range(1, neighbor_window + 1):
            if i + offset < count:
                pairs.append((i, i + offset))
    if max_pairs is not None and len(pairs) > max_pairs:
        stride = int(np.ceil(len(pairs) / max_pairs))
        pairs = pairs[::stride]

    edges: list[dict[str, Any]] = []
    iterator = tqdm(pairs, desc="overlap", unit="pair", disable=not show_progress)
    for i, j in iterator:
        result = pair_overlap(images[i], images[j], method=method, min_inliers=min_inliers)
        if result["connected"]:
            edges.append({"i": i, "j": j, **result})
    log.info("frame graph: %d connected edges across %d frames", len(edges), count)
    return edges
