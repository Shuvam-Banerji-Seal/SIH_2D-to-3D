"""Command-line interface for drone3d."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path

import yaml

from drone3d.config import ALL_STAGES, PipelineConfig, load_config
from drone3d.exceptions import ConfigError, Drone3DError
from drone3d.logging_utils import setup_logging
from drone3d.pipeline import Pipeline
from drone3d.utils.shell import which
from drone3d.version import __version__

_OPTIONAL_MODULES = (
    "torch",
    "transformers",
    "ultralytics",
    "open3d",
    "pycolmap",
    "trimesh",
    "pyproj",
    "laspy",
    "fastapi",
)
_EXTERNAL_BINARIES = ("colmap", "ffmpeg", "ffprobe", "exiftool")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="drone3d",
        description=(
            "Single-pass drone video to accurate, georeferenced 3D models "
            "(SIH problem statement 26158 / NTRO)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"drone3d {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        aliases=["reconstruct"],
        help="Run the reconstruction pipeline",
        description="Run one or more pipeline stages against a run directory.",
    )
    run_parser.add_argument(
        "--config", "-c", type=Path, default=None, help="YAML configuration file"
    )
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key, e.g. --set preprocess.max_frames=300",
    )
    run_parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory")
    run_parser.add_argument(
        "--stages",
        default=None,
        help=f"Comma-separated subset of: {', '.join(ALL_STAGES)}",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Only log warnings and errors")
    run_parser.set_defaults(func=cmd_run)

    init_parser = subparsers.add_parser("init-config", help="Write a default configuration file")
    init_parser.add_argument("path", nargs="?", type=Path, default=Path("configs/default.yaml"))
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing file")
    init_parser.set_defaults(func=cmd_init_config)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check environment, optional extras and external tools"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    subparsers.add_parser("version", help="Print the version").set_defaults(func=cmd_version)
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config, tuple(args.set))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.quiet:
        config.log_level = "WARNING"

    stages = [stage.strip() for stage in args.stages.split(",")] if args.stages else None
    try:
        pipeline = Pipeline(config, args.run_dir)
        setup_logging(
            config.log_level,
            logfile=pipeline.run_dir / "logs" / "run.log",
            quiet=args.quiet,
        )
        result = pipeline.run(stages)
    except Drone3DError as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 1

    width = max((len(stage.name) for stage in result.stages), default=8) + 2
    print(f"run directory: {result.run_dir}")
    for stage in result.stages:
        print(
            f"  {stage.status.upper():8} {stage.name:<{width}} "
            f"{stage.duration_s:7.2f}s  {stage.message}"
        )
    report = result.run_dir / "report.html"
    if report.is_file():
        print(f"report: {report}")
    return 0 if result.ok else 1


def cmd_init_config(args: argparse.Namespace) -> int:
    target: Path = args.path
    if target.exists() and not args.force:
        print(f"refusing to overwrite existing file: {target} (use --force)", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(PipelineConfig().to_dict(), sort_keys=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote default configuration: {target}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"drone3d        {__version__}")
    print(
        f"python         {platform.python_version()} ({platform.system().lower()} {platform.machine()})"
    )
    print(f"executable     {sys.executable}")

    print("\ncore dependencies:")
    for module in ("numpy", "cv2", "yaml", "tqdm"):
        spec = importlib.util.find_spec(module)
        print(f"  {'OK ' if spec else 'MISSING'} {module}")

    print("\noptional backends (install extras via uv sync --extra <name>):")
    for module in _OPTIONAL_MODULES:
        spec = importlib.util.find_spec(module)
        print(f"  {'OK ' if spec else '-- '} {module}")

    print("\nexternal tools:")
    for binary in _EXTERNAL_BINARIES:
        location = which(binary)
        print(f"  {'OK ' if location else '-- '} {binary:<9} {location or '(not on PATH)'}")

    print("\ngpu:")
    try:
        import torch

        print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    except ImportError:
        print("  torch not installed (CPU-only pipeline)")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
