"""Logging helpers shared by the CLI, pipeline and library entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

__all__ = ["get_logger", "setup_logging"]


def setup_logging(
    level: str | int = "INFO",
    logfile: str | Path | None = None,
    *,
    quiet: bool = False,
) -> None:
    """Configure root logging for a pipeline run.

    Args:
        level: Logging level name or numeric value.
        logfile: Optional file that receives a copy of every record.
        quiet: When true, suppress console output (file logging still applies).
    """
    handlers: list[logging.Handler] = []
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stderr))
    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
