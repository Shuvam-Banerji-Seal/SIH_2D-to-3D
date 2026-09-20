"""Backend interfaces for dense reconstruction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from drone3d.config import DenseConfig
from drone3d.exceptions import BackendUnavailable, ConfigError
from drone3d.types import DenseResult, SfMResult

__all__ = ["DenseBackend", "NullDenseBackend", "get_dense_backend"]


class DenseBackend(ABC):
    """Abstract dense reconstruction backend."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the backend's runtime dependencies are present."""

    @abstractmethod
    def reconstruct(
        self,
        images_dir: Path,
        sfm: SfMResult,
        output_dir: Path,
        config: DenseConfig,
    ) -> DenseResult:
        """Produce a dense point cloud (or per-frame depth maps) for a SfM model."""


class NullDenseBackend(DenseBackend):
    """No-op backend used when dense reconstruction is explicitly disabled."""

    name = "none"

    def is_available(self) -> bool:
        return True

    def reconstruct(
        self, images_dir: Path, sfm: SfMResult, output_dir: Path, config: DenseConfig
    ) -> DenseResult:
        raise BackendUnavailable("dense backend is set to 'none'")


def get_dense_backend(name: str, *, binary: str = "colmap") -> DenseBackend:
    """Factory returning a dense backend by name."""
    normalized = name.strip().lower()
    if normalized in {"none", "null", "off"}:
        return NullDenseBackend()
    if normalized == "mvs":
        from drone3d.dense.mvs import ColmapMvsBackend

        return ColmapMvsBackend(binary=binary)
    if normalized == "mono":
        from drone3d.dense.mono_depth import MonoDepthBackend

        return MonoDepthBackend()
    raise ConfigError(f"unknown dense backend '{name}' (expected 'mvs', 'mono' or 'none')")
