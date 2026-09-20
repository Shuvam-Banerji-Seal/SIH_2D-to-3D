"""Backend interfaces for surface extraction and texturing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from drone3d.config import MeshConfig
from drone3d.exceptions import BackendUnavailable, ConfigError
from drone3d.types import MeshResult

__all__ = ["MeshBackend", "NullMeshBackend", "get_mesh_backend"]


class MeshBackend(ABC):
    """Abstract meshing backend."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the backend's runtime dependencies are present."""

    @abstractmethod
    def build(
        self,
        dense_ply: Path,
        images_dir: Path,
        output_dir: Path,
        config: MeshConfig,
    ) -> MeshResult:
        """Extract a surface mesh, and optionally texture it, from a dense cloud."""


class NullMeshBackend(MeshBackend):
    """No-op backend used when meshing is explicitly disabled."""

    name = "none"

    def is_available(self) -> bool:
        return True

    def build(
        self, dense_ply: Path, images_dir: Path, output_dir: Path, config: MeshConfig
    ) -> MeshResult:
        raise BackendUnavailable("mesh backend is set to 'none'")


def get_mesh_backend(name: str, *, binary: str = "colmap") -> MeshBackend:
    """Factory returning a mesh backend by name."""
    normalized = name.strip().lower()
    if normalized in {"none", "null", "off"}:
        return NullMeshBackend()
    if normalized in {"poisson", "delaunay"}:
        from drone3d.mesh.colmap_mesher import ColmapMesher

        return ColmapMesher(method=normalized, binary=binary)
    if normalized == "open3d":
        from drone3d.mesh.texturing import Open3DMesher

        return Open3DMesher()
    raise ConfigError(f"unknown mesh backend '{name}' (expected 'poisson', 'delaunay', 'open3d')")
