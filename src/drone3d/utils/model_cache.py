"""Where model weights are cached.

The machine this project was developed on keeps every model on one shared disk, ``/store/huggingface``.
Anywhere else that disk does not exist, and pointing the libraries at it would make their first download
fail with a permission error, so it is used only when it is there.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["STORE", "hf_hub_cache", "use_weight_store"]

STORE = Path("/store/huggingface")


def use_weight_store() -> None:
    """Cache Hugging Face and torch weights on :data:`STORE` if this machine has it.

    Elsewhere the libraries' own defaults (``~/.cache``) apply. A variable that is already set always wins.
    """
    if STORE.is_dir():
        os.environ.setdefault("HF_HUB_CACHE", str(STORE))
        os.environ.setdefault("TORCH_HOME", str(STORE / "torch"))


def hf_hub_cache() -> Path:
    """Where Hugging Face models are looked up: ``$HF_HUB_CACHE``, else :data:`STORE`, else ``$HF_HOME/hub``."""
    if env := os.environ.get("HF_HUB_CACHE"):
        return Path(env)
    if STORE.is_dir():
        return STORE
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
