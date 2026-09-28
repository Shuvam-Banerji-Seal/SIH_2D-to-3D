"""GPU JPEG encode / decode from two threads at once: the pattern two engine slots produce."""

from __future__ import annotations

import threading

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="nvJPEG needs a GPU")


def test_two_threads_encode_and_decode_on_the_gpu() -> None:
    from drone3d.gpu.nvjpeg import decode_jpeg, encode_jpeg

    errors: list[BaseException] = []

    def work(seed: int) -> None:
        try:
            g = torch.Generator(device="cuda").manual_seed(seed)
            for _ in range(20):
                img = torch.randint(0, 255, (3, 360 + 8 * seed, 640), dtype=torch.uint8, device="cuda", generator=g)
                data = encode_jpeg([img, img], quality=90)
                blobs = [d.numpy().tobytes() for d in data]  # after the call, as the keyframe writer does
                for blob in blobs:  # every bitstream is a complete JPEG
                    assert blob[:2] == b"\xff\xd8" and blob[-2:] == b"\xff\xd9"
                back = decode_jpeg(data[0], device="cuda")
                assert back.shape == img.shape
        except BaseException as exc:  # noqa: BLE001  (report any failure from the thread)
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    torch.cuda.synchronize()  # the context is still healthy
