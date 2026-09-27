"""COLMAP meshing backends (Poisson / Delaunay) with optional texture mapping."""

from __future__ import annotations

from pathlib import Path

from drone3d.config import MeshConfig
from drone3d.exceptions import BackendUnavailable, Drone3DError, ReconstructionError
from drone3d.logging_utils import get_logger
from drone3d.mesh.base import MeshBackend
from drone3d.types import MeshResult
from drone3d.utils.ply import read_ply_header
from drone3d.utils.shell import run_command, which

__all__ = ["ColmapMesher", "ply_element_counts"]

log = get_logger(__name__)


def ply_element_counts(path: str | Path) -> tuple[int, int]:
    """Return ``(vertex_count, face_count)`` from a PLY header, if present."""
    header = read_ply_header(path)
    counts = {element["name"]: element["count"] for element in header["elements"]}
    return int(counts.get("vertex", 0)), int(counts.get("face", 0))


class ColmapMesher(MeshBackend):
    """Runs ``poisson_mesher``/``delaunay_mesher`` and ``texture_mesher``."""

    name = "colmap"

    def __init__(self, method: str = "poisson", binary: str = "colmap") -> None:
        self.method = method
        self.binary = binary

    def is_available(self) -> bool:
        return which(self.binary) is not None

    def build(
        self,
        dense_ply: Path,
        images_dir: Path,
        output_dir: Path,
        config: MeshConfig,
    ) -> MeshResult:
        resolved = which(self.binary)
        if resolved is None:
            raise BackendUnavailable(
                f"COLMAP binary '{self.binary}' not found on PATH (required for meshing)"
            )
        dense_ply = Path(dense_ply)
        if not dense_ply.is_file():
            raise ReconstructionError(f"dense point cloud not found: {dense_ply}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        method = self.method
        mesh_path = output_dir / f"mesh-{method}.ply"
        try:
            self._mesh(resolved, method, dense_ply, mesh_path, config)
        except Drone3DError as exc:
            # Poisson surface trimming is known to crash (SIGSEGV) on some
            # clouds; Delaunay is COLMAP's documented more robust mesher.
            if method != "poisson":
                raise
            log.warning("poisson_mesher failed (%s); falling back to delaunay_mesher", exc)
            method = "delaunay"
            mesh_path = output_dir / f"mesh-{method}.ply"
            self._mesh(resolved, method, dense_ply, mesh_path, config)

        if not mesh_path.is_file():
            raise ReconstructionError(f"meshing produced no output: {mesh_path}")
        vertices, faces = ply_element_counts(mesh_path)

        textured_path: Path | None = None
        if config.texture:
            textured_dir = output_dir / "textured"
            textured_dir.mkdir(parents=True, exist_ok=True)
            # COLMAP 4.x renamed this command: it is `mesh_texturer`, not
            # `texture_mesher` (F25). It takes the undistorter *workspace*
            # (the directory holding `images/` and `sparse/` -- i.e. the dense
            # workspace `dense_ply` lives in) and emits a textured PLY plus an
            # atlas image; there is no `mesh.obj`.
            run_command(
                [
                    resolved,
                    "mesh_texturer",
                    "--workspace_path",
                    dense_ply.parent,
                    "--input_path",
                    mesh_path,
                    "--output_path",
                    textured_dir,
                ]
            )
            # Accept whichever textured artefact the tool produced.
            candidate = next(
                (
                    p
                    for p in sorted(textured_dir.iterdir())
                    if p.suffix.lower() in {".ply", ".obj"} and "textured" in p.stem.lower()
                ),
                None,
            )
            if candidate is None:
                candidate = next(
                    (
                        p
                        for p in sorted(textured_dir.iterdir())
                        if p.suffix.lower() in {".ply", ".obj"}
                    ),
                    None,
                )
            textured_path = candidate

        log.info("mesh: %d vertices / %d faces -> %s", vertices, faces, mesh_path)
        return MeshResult(
            backend=f"colmap-{method}",
            mesh_path=mesh_path,
            textured_mesh_path=textured_path,
            num_vertices=vertices,
            num_faces=faces,
            metadata={"textured": textured_path is not None},
        )

    def _mesh(
        self,
        resolved: str,
        method: str,
        dense_ply: Path,
        mesh_path: Path,
        config: MeshConfig,
    ) -> None:
        """Run one mesher; raises on failure so the caller can fall back."""
        if method == "poisson":
            run_command(
                [
                    resolved,
                    "poisson_mesher",
                    "--input_path",
                    dense_ply,
                    "--output_path",
                    mesh_path,
                    "--PoissonMeshing.depth",
                    str(config.depth),
                    "--PoissonMeshing.trim",
                    str(config.trim),
                ]
            )
        else:
            # `delaunay_mesher` has no `--output_type` option (COLMAP 4.x rejects
            # it outright); the output format is inferred from the file extension.
            run_command(
                [
                    resolved,
                    "delaunay_mesher",
                    "--input_path",
                    dense_ply.parent,
                    "--output_path",
                    mesh_path,
                ]
            )
