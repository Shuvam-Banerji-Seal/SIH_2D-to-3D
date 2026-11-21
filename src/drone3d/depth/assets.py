"""Where the Marigold v2 weights live.

Kept free of ``torch`` so ``drone3d doctor`` and the engine's capability probe can look for the weights
on an install that has not got the ``depth`` extra.
"""

from __future__ import annotations

import os
from pathlib import Path

from drone3d.utils.model_cache import STORE

__all__ = ["DEFAULT_ASSETS", "resolve_assets_dir"]

DEFAULT_ASSETS = STORE / "marigold-v2"


def resolve_assets_dir(override: str | Path | None = None) -> Path:
    """``override``, else ``$DEPTH_ASSETS_DIR``, else :data:`DEFAULT_ASSETS`."""
    return Path(override or os.environ.get("DEPTH_ASSETS_DIR", DEFAULT_ASSETS))
