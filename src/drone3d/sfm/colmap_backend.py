"""COLMAP command-line backend (feature extraction -> matching -> mapping)."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from drone3d.config import SfMConfig
from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.logging_utils import get_logger
from drone3d.sfm.base import SfMBackend
from drone3d.types import SfMResult
from drone3d.utils.shell import run_command, which

__all__ = ["ColmapSfMBackend"]

log = get_logger(__name__)

_MATCHERS = {
    "sequential": "sequential_matcher",
    "exhaustive": "exhaustive_matcher",
    "vocab_tree": "vocab_tree_matcher",
}


class ColmapSfMBackend(SfMBackend):
    """Wraps ``colmap feature_extractor``, a matcher and ``colmap mapper``."""

    name = "colmap"

    def __init__(self, binary: str = "colmap") -> None:
        self.binary = binary

    def is_available(self) -> bool:
        return which(self.binary) is not None

    def _executable(self) -> str:
        resolved = which(self.binary)
        if resolved is None:
            raise BackendUnavailable(
                f"COLMAP binary '{self.binary}' not found on PATH. "
                "Install COLMAP (https://colmap.github.io/) or set sfm.binary."
            )
        return resolved

    def reconstruct(
        self,
        images_dir: Path,
        output_dir: Path,
        config: SfMConfig,
        mask_dir: Path | None = None,
    ) -> SfMResult:
        binary = self._executable()
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        database = output_dir / "database.db"
        if database.exists():
            database.unlink()

        self._extract_features(binary, images_dir, database, config, mask_dir=mask_dir)
        self._match(binary, database, config)

        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                binary,
                "mapper",
                "--database_path",
                database,
                "--image_path",
                images_dir,
                "--output_path",
                sparse_dir,
                *config.extra_args,
            ]
        )

        model_path = _largest_model(sparse_dir)
        if model_path is None:
            raise ReconstructionError(
                "COLMAP mapper produced no model; the pass may have too little overlap"
            )
        log.info("COLMAP registered model: %s", model_path)

        ply_path = output_dir / "sparse.ply"
        run_command(
            [
                binary,
                "model_converter",
                "--input_path",
                model_path,
                "--output_path",
                ply_path,
                "--output_type",
                "PLY",
            ]
        )
        text_dir = output_dir / "sparse_txt"
        # `model_converter --output_type TXT` writes several files *into* a
        # directory and aborts if it does not already exist (PLY output does not
        # have this requirement, which is why the step above needs no mkdir).
        text_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                binary,
                "model_converter",
                "--input_path",
                model_path,
                "--output_path",
                text_dir,
                "--output_type",
                "TXT",
            ]
        )
        stats = _read_model_stats(text_dir)
        return SfMResult(
            backend=self.name,
            model_path=model_path,
            sparse_ply=ply_path,
            num_registered_images=stats["num_images"],
            num_points=stats["num_points"],
            mean_reprojection_error_px=stats["mean_reprojection_error"],
            metadata={"database": str(database), "text_model": str(text_dir)},
        )

    def _extract_features(
        self,
        binary: str,
        images_dir: Path,
        database: Path,
        config: SfMConfig,
        mask_dir: Path | None = None,
    ) -> None:
        cmd: list[str | Path] = [
            binary,
            "feature_extractor",
            "--database_path",
            database,
            "--image_path",
            images_dir,
            "--ImageReader.camera_model",
            config.camera_model,
            "--ImageReader.single_camera",
            "1" if config.single_camera else "0",
            "--SiftExtraction.max_num_features",
            str(config.max_features),
        ]
        # COLMAP masks: the file for `image_path/abc/012.jpg` must live at
        # `mask_path/abc/012.jpg.png`, and regions with value 0 are ignored.
        if mask_dir is not None and any(Path(mask_dir).glob("*.png")):
            cmd += ["--ImageReader.mask_path", Path(mask_dir)]
        run_command([str(item) for item in [*cmd, *config.extra_args]])

    def _match(self, binary: str, database: Path, config: SfMConfig) -> None:
        matcher = _MATCHERS.get(config.matcher)
        if matcher is None:
            raise ReconstructionError(f"unsupported COLMAP matcher '{config.matcher}'")
        run_command([binary, matcher, "--database_path", database, *config.extra_args])


def _largest_model(sparse_dir: Path) -> Path | None:
    """Return the sub-model with the most registered images (COLMAP names them 0, 1, ...)."""
    candidates = [
        path
        for path in sparse_dir.iterdir()
        if path.is_dir() and ((path / "images.bin").exists() or (path / "images.txt").exists())
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            (path / "images.bin").stat().st_size
            if (path / "images.bin").exists()
            else (path / "images.txt").stat().st_size
        ),
    )


class _ModelStats(TypedDict):
    """Counts parsed out of a COLMAP TXT model."""

    num_images: int
    num_points: int
    mean_reprojection_error: float


def _read_model_stats(text_dir: Path) -> _ModelStats:
    """Parse image count, point count and mean reprojection error from a TXT model."""
    stats: _ModelStats = {
        "num_images": 0,
        "num_points": 0,
        "mean_reprojection_error": 0.0,
    }
    images_txt = text_dir / "images.txt"
    if images_txt.is_file():
        lines = [
            line for line in images_txt.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        stats["num_images"] = len(lines) // 2

    points_txt = text_dir / "points3D.txt"
    if points_txt.is_file():
        errors: list[float] = []
        for line in points_txt.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                errors.append(float(fields[7]))
            except ValueError:
                continue
        stats["num_points"] = len(errors)
        stats["mean_reprojection_error"] = round(sum(errors) / len(errors), 4) if errors else 0.0
    return stats
