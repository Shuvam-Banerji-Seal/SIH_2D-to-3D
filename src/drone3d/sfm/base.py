"""Backend interfaces for structure-from-motion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from drone3d.config import SfMConfig
from drone3d.exceptions import BackendUnavailable, ConfigError
from drone3d.types import SfMResult

__all__ = ["NullSfMBackend", "SfMBackend", "get_sfm_backend"]


class SfMBackend(ABC):
    """Abstract incremental structure-from-motion backend."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the backend's runtime dependencies are present."""

    @abstractmethod
    def reconstruct(self, images_dir: Path, output_dir: Path, config: SfMConfig) -> SfMResult:
        """Reconstruct camera poses and a sparse point cloud from calibrated images."""

    def describe(self) -> str:
        return f"{self.name} (available={self.is_available()})"


class NullSfMBackend(SfMBackend):
    """No-op backend used when SfM is explicitly disabled (e.g. ingest-only runs)."""

    name = "none"

    def is_available(self) -> bool:
        return True

    def reconstruct(self, images_dir: Path, output_dir: Path, config: SfMConfig) -> SfMResult:
        raise BackendUnavailable(
            "SfM backend is set to 'none'; set sfm.backend=colmap to reconstruct a model"
        )


def get_sfm_backend(name: str, binary: str = "colmap") -> SfMBackend:
    """Factory returning an SfM backend by name.

    Raises:
        ConfigError: If the backend name is unknown.
    """
    normalized = name.strip().lower()
    if normalized in {"none", "null", "off"}:
        return NullSfMBackend()
    if normalized == "colmap":
        from drone3d.sfm.colmap_backend import ColmapSfMBackend

        return ColmapSfMBackend(binary=binary)
    raise ConfigError(f"unknown SfM backend '{name}' (expected 'colmap' or 'none')")
