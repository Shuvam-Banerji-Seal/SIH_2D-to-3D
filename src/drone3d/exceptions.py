"""Exception hierarchy for the drone3d pipeline."""

from __future__ import annotations

__all__ = [
    "BackendUnavailable",
    "ConfigError",
    "Drone3DError",
    "IngestionError",
    "ReconstructionError",
]


class Drone3DError(Exception):
    """Base class for all errors raised by :mod:`drone3d`."""


class ConfigError(Drone3DError):
    """Raised when a configuration file or override is invalid."""


class IngestionError(Drone3DError):
    """Raised when video or telemetry input cannot be read."""


class BackendUnavailable(Drone3DError):
    """Raised when an optional reconstruction backend is not installed."""


class ReconstructionError(Drone3DError):
    """Raised when a reconstruction stage fails."""
