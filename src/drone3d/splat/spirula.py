"""Drive spirula-studio (``third_party/spirula-studio``): SfM, 3DGS training, meshing.

spirula-studio is GPLv3 and is used strictly as a separate program: this
module builds command lines, runs the binary, and reads the files and log lines
it leaves behind. The binary is looked up in ``$DRONE3D_SPIRULA``, then the
submodule's ``build_vulkan`` / ``build_cuda`` trees, then ``PATH``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from drone3d.exceptions import BackendUnavailable, ReconstructionError
from drone3d.logging_utils import get_logger

__all__ = [
    "MeshResult",
    "SfmRunResult",
    "TrainResult",
    "run_mesh",
    "run_sfm",
    "run_train",
    "spirula_binary",
]

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def spirula_binary() -> Path:
    """Locate the ``spirula`` executable."""
    candidates = []
    if env := os.environ.get("DRONE3D_SPIRULA"):
        candidates.append(Path(env))
    submodule = _REPO_ROOT / "third_party" / "spirula-studio"
    candidates += [submodule / "build_vulkan" / "spirula", submodule / "build_cuda" / "spirula"]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    found = shutil.which("spirula")
    if found:
        return Path(found)
    raise BackendUnavailable(
        "spirula binary not found; build it with "
        "`bash third_party/spirula-studio/build_develop.bash -DSS_BACKEND=vulkan -DSS_BUILD_GUI=OFF`"
    )


STALLED = -1000  # _run's return code when the process stopped writing and was killed


def _run(
    cmd: Sequence[str | Path], log_path: Path, cwd: Path | None = None, *, stall_s: float | None = None
) -> tuple[int, str, float]:
    """Run ``cmd`` streaming stdout+stderr to ``log_path``; return (code, text, seconds).

    With ``stall_s``, a process whose log has not grown for that long is killed
    and the code is :data:`STALLED`: a Vulkan trainer once sat in ``poll`` at step 1
    for 20 minutes (6 s of CPU), and with no limit it held the engine's queue.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = [str(c) for c in cmd]
    log.info("exec: %s", " ".join(args))
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(args) + "\n")
        handle.flush()
        process = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, cwd=cwd)
        size, quiet_since = -1, time.monotonic()
        while True:
            try:
                code = process.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                pass
            now_size = log_path.stat().st_size
            if now_size != size:
                size, quiet_since = now_size, time.monotonic()
            elif stall_s is not None and time.monotonic() - quiet_since > stall_s:
                process.kill()
                process.wait()
                handle.write(f"\n[drone3d] no output for {stall_s:.0f} s: killed\n")
                code = STALLED
                break
    elapsed = time.perf_counter() - started
    return code, log_path.read_text(encoding="utf-8", errors="replace"), elapsed


# ----------------------------------------------------------------------------- SfM


@dataclass
class SfmRunResult:
    workspace: Path
    models: list[dict]  # per sparse/N: {"path", "images", "points", "mean_reprojection_px"}
    registered_images: int
    input_images: int
    mean_reprojection_px: float | None
    result_line: str
    seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
    log_path: Path | None = None

    @property
    def best_model(self) -> Path | None:
        return Path(self.models[0]["path"]) if self.models else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["workspace"] = str(self.workspace)
        d["log_path"] = str(self.log_path) if self.log_path else None
        d["best_model"] = str(self.best_model) if self.best_model else None
        return d


def _hms(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _model_stats(model_dir: Path) -> dict:
    try:
        import pycolmap
    except ImportError:  # pragma: no cover - pycolmap is in the sfm extra
        return {
            "path": str(model_dir),
            "images": None,
            "points": None,
            "mean_reprojection_px": None,
        }
    rec = pycolmap.Reconstruction(str(model_dir))
    errors = [p.error for p in rec.points3D.values() if p.error >= 0]
    return {
        "path": str(model_dir),
        "images": int(sum(1 for im in rec.images.values() if im.has_pose)),
        "points": int(len(rec.points3D)),
        "mean_reprojection_px": round(float(sum(errors) / len(errors)), 4) if errors else None,
        "mean_track_length": round(float(rec.compute_mean_track_length()), 3)
        if rec.points3D
        else None,
    }


def run_sfm(
    dataset: Path,
    *,
    sequences: Sequence[str] = (),
    quality: str = "high",
    data_type: str = "video",
    camera_mode: str = "folder",
    camera_model: str = "radial",
    features: str = "sift",
    extra: Sequence[str] = (),
    log_path: Path | None = None,
) -> SfmRunResult:
    """``spirula sfm auto`` over ``dataset/images`` into ``dataset/sparse/N``.

    ``sequences`` names image sub-folders shot in file-name order (one per
    pass), so neighbours are matched and trusted first.
    """
    binary = spirula_binary()
    images = dataset / "images"
    if not images.is_dir():
        raise ReconstructionError(f"no images/ under {dataset}")
    sparse = dataset / "sparse"
    for stale in (sparse, dataset / "features", dataset / "matches.bin"):
        if stale.is_dir():
            shutil.rmtree(stale)  # never reconstruct on top of a stale run
        elif stale.exists():
            stale.unlink()
    cmd: list[str | Path] = [binary, "sfm", "auto", dataset, "-o", dataset,
                             "--data-type", data_type, "--quality", quality,
                             "--camera-mode", camera_mode, "--camera-model", camera_model,
                             "--features", features]  # fmt: skip
    if sequences:
        cmd += ["--sequence", ",".join(sequences)]
    cmd += list(extra)
    log_path = log_path or dataset / "sfm.log"
    code, text, seconds = _run(cmd, log_path)
    if code not in (0, 3):  # 3 = reconstructed, but only partially
        raise ReconstructionError(f"spirula sfm failed (exit {code}); see {log_path}")
    models = sorted(
        (_model_stats(p) for p in sparse.iterdir() if (p / "images.bin").is_file() or (p / "images.txt").is_file()),
        key=lambda m: -(m["images"] or 0),
    ) if sparse.is_dir() else []  # fmt: skip
    stage = {}
    for name in ("Extraction", "Matching", "Mapping", "Assembly", "Total"):
        if m := re.search(rf"\[run\]\s+{name}:\s+(\d+:\d+:\d+\.\d+)", text):
            stage[name.lower()] = _hms(m.group(1))
    reproj = re.search(r"Reprojection error: mean ([\d.]+) px", text)
    result = re.search(r"RESULT: (.*)", text)
    n_in = sum(1 for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return SfmRunResult(
        workspace=dataset,
        models=models,
        registered_images=sum(m["images"] or 0 for m in models),
        input_images=n_in,
        mean_reprojection_px=float(reproj.group(1)) if reproj else None,
        result_line=result.group(1).strip() if result else "",
        seconds=seconds,
        stage_seconds=stage,
        log_path=log_path,
    )


# --------------------------------------------------------------------------- train


@dataclass
class TrainResult:
    run_dir: Path
    splat_ply: Path | None
    num_splats: int | None
    steps: int
    seconds: float
    eval_metrics: dict
    config: dict
    log_path: Path

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "splat_ply": str(self.splat_ply) if self.splat_ply else None,
            "num_splats": self.num_splats,
            "steps": self.steps,
            "seconds": round(self.seconds, 2),
            "eval_metrics": self.eval_metrics,
            "log_path": str(self.log_path),
        }


def dense_init_model(model_dir: Path, points_ply: Path, out_dir: Path, *, max_points: int = 400_000) -> Path:
    """A copy of an SfM model whose 3D points are the dense cloud, as the splats' starting points.

    Flow SfM keeps ~12 tracks per image (Jal Mahal's largest pass: ~1000
    points), which 7k steps cannot densify into a full scene; the TSDF cloud
    covers every surface the depth maps saw. Cameras and images are copied
    unchanged; the points carry colour and no tracks.
    """
    import open3d as o3d
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    for pid in list(rec.points3D.keys()):
        rec.delete_point3D(pid)
    cloud = o3d.io.read_point_cloud(str(points_ply))
    xyz = np.asarray(cloud.points, dtype=np.float64)
    rgb = (np.asarray(cloud.colors) * 255).round().astype(np.uint8) if cloud.has_colors() else np.full((len(xyz), 3), 128, np.uint8)
    if len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    track = pycolmap.Track()
    for p, c in zip(xyz, rgb, strict=True):
        rec.add_point3D(p, track, c)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    return out_dir


def run_train(
    dataset: Path,
    out_dir: Path,
    *,
    preset: str = "3dgs",
    recon_dir: str | None = None,
    iterations: int = 30000,
    quality: str = "high",
    depth_weight: float = 0.0,
    eval_interval: int = 8,
    flags: dict[str, object] | None = None,
    stall_s: float = 300.0,
) -> TrainResult:
    """``spirula train`` headless; held-out views every ``eval_interval``-th image."""
    binary = spirula_binary()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd: list[str | Path] = [
        binary, "train", preset, "--data", dataset,
        "--output-dir-prefix", out_dir.parent, "--output-dir-name", out_dir.name,
        "--num-iterations", str(iterations), "--quality", quality,
        "--disable-viewer", "1", "--keep-viewer-alive", "0",
        "--eval-mode", "interval", "--eval-interval", str(eval_interval),
        "--save-eval-images", "1",
        "--depth-supervision-weight", str(depth_weight),
    ]  # fmt: skip
    if recon_dir:
        cmd += ["--colmap-recon-dir", recon_dir]
    for key, value in (flags or {}).items():
        cmd += [f"--{key.replace('_', '-')}", str(int(value) if isinstance(value, bool) else value)]
    log_path = out_dir.parent / f"{out_dir.name}.log"
    for attempt in (1, 2):  # a stalled trainer is retried once in a fresh process
        code, text, seconds = _run(cmd, log_path, stall_s=stall_s)
        if code != STALLED:
            break
        log.warning("spirula train %s stalled (attempt %d); %s", out_dir.name, attempt, "retrying" if attempt == 1 else "giving up")
        if out_dir.exists():
            shutil.rmtree(out_dir)
    if code == STALLED:
        raise ReconstructionError(f"spirula train stalled twice (no output for {stall_s:.0f} s); see {log_path}")
    if code != 0:
        raise ReconstructionError(f"spirula train failed (exit {code}); see {log_path}")
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    plys = sorted(out_dir.rglob("splat*.ply"), key=lambda p: p.stat().st_mtime)
    splats = [int(m.group(1)) for m in re.finditer(r"splats (\d+)", text)]
    steps = [int(m.group(1)) for m in re.finditer(r"step\s+(\d+)/", text)]
    return TrainResult(
        run_dir=out_dir,
        splat_ply=plys[-1] if plys else None,
        num_splats=splats[-1] if splats else None,
        steps=steps[-1] if steps else 0,
        seconds=seconds,
        eval_metrics=metrics,
        config=config,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------- mesh


@dataclass
class MeshResult:
    files: list[Path]
    seconds: float
    log_path: Path

    def to_dict(self) -> dict:
        return {
            "files": [str(f) for f in self.files],
            "seconds": round(self.seconds, 2),
            "log_path": str(self.log_path),
        }


def _finished_mesh(run_dir: Path, log_path: Path, formats: Sequence[str]) -> MeshResult | None:
    """The mesh an earlier ``spirula mesh`` made from the current splat, if complete."""
    done = re.search(r"\[meshing\] done:.*\(total ([\d.]+)s\)", log_path.read_text()) if log_path.is_file() else None
    splats = list(run_dir.rglob("splat.ply"))
    if done is None or not splats:
        return None
    newest_splat = max(p.stat().st_mtime for p in splats)
    files = sorted(p for p in run_dir.rglob("mesh*") if p.is_file() and p.stat().st_mtime > newest_splat)
    if not files or any(not any(f.suffix == f".{fmt}" for f in files) for fmt in formats):
        return None
    return MeshResult(files=files, seconds=float(done.group(1)), log_path=log_path)


def run_mesh(
    run_dir: Path,
    *,
    formats: Sequence[str] = ("ply", "glb", "obj"),
    colors: Sequence[str] = ("vertex", "texture"),
    flags: dict[str, object] | None = None,
    reuse: bool = True,
) -> MeshResult:
    """``spirula mesh`` on a finished training run; outputs land beside its splat.ply.

    With ``reuse``, a mesh already produced from the current splat (its log
    reports completion and every requested format is newer than splat.ply) is
    returned instead of meshing again, which takes ~40 min for a 3M-splat model.
    """
    log_path = run_dir.parent / f"{run_dir.name}_mesh.log"
    if reuse and (done := _finished_mesh(run_dir, log_path, formats)) is not None:
        log.info("reusing the mesh of %s (%d files)", run_dir.name, len(done.files))
        return done
    binary = spirula_binary()
    before = {p: p.stat().st_mtime for p in run_dir.rglob("mesh*")}
    cmd: list[str | Path] = [
        binary,
        "mesh",
        run_dir,
        "--format",
        ",".join(formats),
        "--color",
        ",".join(colors),
    ]
    for key, value in (flags or {}).items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    code, _, seconds = _run(cmd, log_path)
    if code != 0:
        raise ReconstructionError(f"spirula mesh failed (exit {code}); see {log_path}")
    files = sorted(
        p for p in run_dir.rglob("mesh*") if p.is_file() and before.get(p) != p.stat().st_mtime
    )
    return MeshResult(files=files, seconds=seconds, log_path=log_path)
