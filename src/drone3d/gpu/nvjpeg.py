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


def _settle(out: Any, *, to_cpu: bool) -> Any:
    """Wait for nvJPEG's own stream, then hand back tensors no later call can overwrite."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # the encoder / decoder run on their own stream, not the current one
    one = not isinstance(out, list)
    items = [out] if one else out
    items = [t.cpu().clone() if to_cpu else t.clone() for t in items]
    return items[0] if one else items


def encode_jpeg(images: Any, **kw: Any) -> Any:
    """``torchvision.io.encode_jpeg`` under the process-wide nvJPEG lock; the bitstreams come back on the CPU.

    Copying them after the lock was released read bitstreams the other slot's next encode was
    overwriting: 48 of 128 keyframes of a two-slot run were corrupt on disk.
    """
    from torchvision.io import encode_jpeg as _encode

    with _LOCK:
        return _settle(_encode(images, **kw), to_cpu=True)


def decode_jpeg(data: Any, **kw: Any) -> Any:
    """``torchvision.io.decode_jpeg`` under the process-wide nvJPEG lock (only GPU decodes need it)."""
    from torchvision.io import decode_jpeg as _decode

    if str(kw.get("device", "cpu")).startswith("cuda"):
        with _LOCK:
            return _settle(_decode(data, **kw), to_cpu=False)
    return _decode(data, **kw)
