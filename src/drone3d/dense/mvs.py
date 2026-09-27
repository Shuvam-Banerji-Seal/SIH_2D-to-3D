"""COLMAP multi-view-stereo dense reconstruction backend."""

from __future__ import annotations

from pathlib import Path

from drone3d.config import DenseConfig
from drone3d.dense.base import DenseBackend
from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.logging_utils import get_logger
from drone3d.types import DenseResult, SfMResult
from drone3d.utils.ply import ply_vertex_count
from drone3d.utils.shell import run_command, which

__all__ = ["ColmapMvsBackend"]

log = get_logger(__name__)


class ColmapMvsBackend(DenseBackend):
    """Runs ``image_undistorter`` -> ``patch_match_stereo`` -> ``stereo_fusion``."""

    name = "mvs"

    def __init__(self, binary: str = "colmap") -> None:
        self.binary = binary

    def is_available(self) -> bool:
        return which(self.binary) is not None

    def reconstruct(
        self,
        images_dir: Path,
        sfm: SfMResult,
        output_dir: Path,
        config: DenseConfig,
    ) -> DenseResult:
        resolved = which(self.binary)
        if resolved is None:
            raise BackendUnavailable(
                f"COLMAP binary '{self.binary}' not found on PATH (required for dense MVS)"
            )
        if sfm.model_path is None:
            raise ReconstructionError("dense MVS requires a sparse model; run the SfM stage first")

        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        dense_dir = output_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                resolved,
                "image_undistorter",
                "--image_path",
                images_dir,
                "--input_path",
                sfm.model_path,
                "--output_path",
                dense_dir,
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(config.max_image_size),
            ]
        )
        run_command(
            [
                resolved,
                "patch_match_stereo",
                "--workspace_path",
                dense_dir,
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true" if config.geom_consistency else "false",
            ]
        )

        fused = dense_dir / "fused.ply"
        run_command(
            [
                resolved,
                "stereo_fusion",
                "--workspace_path",
                dense_dir,
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric" if config.geom_consistency else "photometric",
                "--output_path",
                fused,
            ]
        )
        if not fused.is_file():
            raise ReconstructionError("stereo fusion produced no fused.ply")

        count = ply_vertex_count(fused)
        log.info("dense cloud: %d points -> %s", count, fused)
        return DenseResult(
            backend=self.name,
            fused_ply=fused,
            num_points=count,
            metadata={"dense_dir": str(dense_dir)},
        )
