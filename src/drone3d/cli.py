"""Command-line interface for drone3d."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path

import yaml

from drone3d.config import ALL_STAGES, PipelineConfig, load_config
from drone3d.exceptions import ConfigError, Drone3DError
from drone3d.logging_utils import setup_logging
from drone3d.pipeline import Pipeline
from drone3d.version import __version__

_OPTIONAL_MODULES = (
    ("torch", "gpu"),
    ("torchvision", "gpu"),
    ("pynvml", "gpu"),
    ("diffusers", "depth"),
    ("bitsandbytes", "depth"),
    ("peft", "depth"),
    ("pycolmap", "sfm"),
    ("gsplat", "render"),
    ("open3d", "mesh"),
    ("trimesh", "mesh"),
    ("pyproj", "geo"),
)


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
        help="Override any config key, e.g. --set splat.iterations=7000",
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

    view_parser = subparsers.add_parser("view", help="Serve a run's 3D viewer (export stage) over HTTP")
    view_parser.add_argument("run_dir", help="Run directory containing export/index.html")
    view_parser.add_argument("--host", default="127.0.0.1")
    view_parser.add_argument("--port", type=int, default=8765)
    view_parser.set_defaults(func=cmd_view)

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
    for module in ("numpy", "cv2", "yaml"):
        spec = importlib.util.find_spec(module)
        print(f"  {'OK ' if spec else 'MISSING'} {module}")

    print("\noptional extras (uv sync --extra <name>):")
    for module, extra in _OPTIONAL_MODULES:
        spec = importlib.util.find_spec(module)
        print(f"  {'OK ' if spec else '-- '} {module:<13} [{extra}]")

    print("\nexternal tools:")
    try:
        from drone3d.io.nvdec import ffmpeg_bin

        ff = ffmpeg_bin()
        import subprocess

        version = subprocess.run(
            [ff, "-hide_banner", "-version"], capture_output=True, text=True
        ).stdout.split("\n")[0]
        hw = subprocess.run(
            [ff, "-hide_banner", "-hwaccels"], capture_output=True, text=True
        ).stdout
        print(
            f"  OK  ffmpeg    {ff} ({version.split(' Copyright')[0]}; cuda hwaccel: {'cuda' in hw})"
        )
    except Drone3DError as exc:
        print(f"  --  ffmpeg    {exc}")
    try:
        from drone3d.splat.spirula import spirula_binary

        print(f"  OK  spirula   {spirula_binary()}")
    except Drone3DError as exc:
        print(f"  --  spirula   {str(exc).splitlines()[0]}")
    from drone3d.depth.marigold import _DEFAULT_ASSETS

    assets = Path(os.environ.get("DEPTH_ASSETS_DIR", _DEFAULT_ASSETS)) / "checkpoints"
    ok = (assets / "Qwen-Image-Edit-2509" / "transformer").is_dir() and (
        assets / "Marigold-V2"
    ).is_dir()
    print(f"  {'OK ' if ok else '-- '} marigold  {assets}")

    print("\ngpu:")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(
                f"  torch {torch.__version__} (CUDA {torch.version.cuda}): {props.name}, {props.total_memory / 2**30:.0f} GiB"
            )
        else:
            print(
                f"  torch {torch.__version__}: no CUDA device (keyframes, depth and splats need one)"
            )
    except ImportError:
        print("  torch not installed (uv sync --extra gpu)")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    import functools
    import http.server
    from pathlib import Path

    root = Path(args.run_dir) / "export"
    if not (root / "index.html").is_file():
        print(f"no viewer in {root} (run the export stage)", file=sys.stderr)
        return 1
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.func.extensions_map.update(  # type: ignore[attr-defined]
        {".glb": "model/gltf-binary", ".ply": "application/octet-stream", ".las": "application/octet-stream",
         ".fbx": "application/octet-stream", ".tif": "image/tiff", ".js": "text/javascript"}
    )  # fmt: skip
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"viewer: http://{args.host}:{args.port}/  (Ctrl+C to stop)")
        httpd.serve_forever()
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
