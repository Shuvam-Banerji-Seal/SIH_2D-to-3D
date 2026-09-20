"""Open3D/Trimesh meshing and export helpers (optional ``mesh`` extra)."""

from __future__ import annotations

from pathlib import Path

from drone3d.config import MeshConfig
from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.logging_utils import get_logger
from drone3d.mesh.base import MeshBackend
from drone3d.types import MeshResult

__all__ = [
    "Open3DMesher",
    "convert_mesh_format",
    "has_open3d",
    "has_trimesh",
    "poisson_mesh_from_cloud",
]

log = get_logger(__name__)


def has_open3d() -> bool:
    try:
        import open3d  # noqa: F401
    except ImportError:
        return False
    return True


def has_trimesh() -> bool:
    try:
        import trimesh  # noqa: F401
    except ImportError:
        return False
    return True


def poisson_mesh_from_cloud(
    cloud_path: str | Path,
    mesh_path: str | Path,
    *,
    depth: int = 11,
    trim: float = 10.0,
    density_quantile: float = 0.02,
) -> Path:
    """Screened-Poisson surface reconstruction with density-based trimming."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BackendUnavailable(
            "Open3D meshing needs the 'mesh' extra: uv sync --extra mesh"
        ) from exc

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if len(cloud.points) == 0:
        raise ReconstructionError(f"point cloud is empty: {cloud_path}")
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    cloud.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    if 0.0 < density_quantile < 1.0 and len(densities) > 0:
        import numpy as np

        threshold = float(np.quantile(np.asarray(densities), density_quantile))
        mesh.remove_vertices_by_mask(np.asarray(densities) < threshold)

    target = Path(mesh_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(target), mesh):
        raise ReconstructionError(f"failed to write mesh: {target}")
    log.info("open3d poisson mesh: %d vertices -> %s", len(mesh.vertices), target)
    return target


def convert_mesh_format(
    mesh_path: str | Path,
    out_path: str | Path,
    *,
    target_faces: int | None = None,
) -> Path:
    """Convert/decimate a mesh using Trimesh (supports OBJ, PLY, GLB, STL)."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BackendUnavailable(
            "mesh conversion needs the 'mesh' extra: uv sync --extra mesh"
        ) from exc

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if target_faces is not None and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(target))
    return target


class Open3DMesher(MeshBackend):
    """Pure-Python meshing path that does not require a COLMAP binary."""

    name = "open3d"

    def is_available(self) -> bool:
        return has_open3d()

    def build(
        self,
        dense_ply: Path,
        images_dir: Path,
        output_dir: Path,
        config: MeshConfig,
    ) -> MeshResult:
        if not self.is_available():
            raise BackendUnavailable("Open3D meshing needs the 'mesh' extra: uv sync --extra mesh")
        output_dir = Path(output_dir)
        mesh_path = poisson_mesh_from_cloud(
            dense_ply, output_dir / "mesh-open3d.ply", depth=config.depth, trim=config.trim
        )
        vertices = faces = 0
        textured_path = None
        if has_trimesh():
            from drone3d.mesh.colmap_mesher import ply_element_counts

            vertices, faces = ply_element_counts(mesh_path)
            textured_path = convert_mesh_format(
                mesh_path, output_dir / "mesh.glb", target_faces=config.target_faces
            )
        return MeshResult(
            backend=self.name,
            mesh_path=mesh_path,
            textured_mesh_path=textured_path,
            num_vertices=vertices,
            num_faces=faces,
            metadata={"has_vertex_colors": True},
        )
