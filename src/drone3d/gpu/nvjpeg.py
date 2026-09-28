"""GPU JPEG encode / decode, one call at a time per process.

torchvision's CUDA JPEG encoder and decoder each keep one global nvJPEG state; two engine
slots writing keyframes at once corrupted it (a device-side assert that poisoned the CUDA
context, then errors 710 in every later call). A call is milliseconds, so serialising them
costs nothing measurable.
"""

from __future__ import annotations

import threading
from typing import Any

__all__ = ["decode_jpeg", "encode_jpeg"]

_LOCK = threading.Lock()


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()  # the shared state is free before another thread takes it


def encode_jpeg(images: Any, **kw: Any) -> Any:
    """``torchvision.io.encode_jpeg`` under the process-wide nvJPEG lock."""
    from torchvision.io import encode_jpeg as _encode

    with _LOCK:
        out = _encode(images, **kw)
        _sync()
    return out


def decode_jpeg(data: Any, **kw: Any) -> Any:
    """``torchvision.io.decode_jpeg`` under the process-wide nvJPEG lock (only GPU decodes need it)."""
    from torchvision.io import decode_jpeg as _decode

    if str(kw.get("device", "cpu")).startswith("cuda"):
        with _LOCK:
            out = _decode(data, **kw)
            _sync()
        return out
    return _decode(data, **kw)
