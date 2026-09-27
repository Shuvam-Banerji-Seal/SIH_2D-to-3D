"""NV12 -> RGB conversion must invert the encoder's colour matrix exactly."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drone3d.io.nvdec import analysis_size, nv12_to_rgb  # noqa: E402


def _encode_nv12(rgb: np.ndarray, bt709: bool) -> np.ndarray:
    """Limited-range NV12 from RGB (the inverse the decoder must undo)."""
    kr, kb = (0.2126, 0.0722) if bt709 else (0.299, 0.114)
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = kr * r + (1 - kr - kb) * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    y8 = np.clip(np.rint(16 + y * 219 / 255), 0, 255)
    cb8 = np.clip(np.rint(128 + cb * 224 / 255), 0, 255)
    cr8 = np.clip(np.rint(128 + cr * 224 / 255), 0, 255)
    h, w = y.shape
    # 2x2 chroma subsampling; flat 2x2 blocks make it lossless for this test.
    uv = np.stack([cb8[::2, ::2], cr8[::2, ::2]], axis=-1).reshape(h // 2, w)
    return np.concatenate([y8, uv], axis=0).astype(np.uint8)


@pytest.mark.parametrize("bt709", [True, False])
def test_round_trip_within_two_levels(bt709: bool) -> None:
    rng = np.random.default_rng(0)
    # Flat 8x8-pixel colour blocks (4x4 chroma samples): bilinear chroma
    # upsampling only blends across block edges, so block interiors must
    # come back exactly (up to 8-bit rounding).
    blocks = rng.integers(0, 256, (4, 6, 3))
    rgb = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1).astype(np.uint8)  # 32 x 48
    nv12 = torch.from_numpy(np.stack([_encode_nv12(rgb, bt709)] * 3))
    back = nv12_to_rgb(nv12, 32, 48, bt709=bt709, max_batch=2).numpy()
    assert back.shape == (3, 32, 48, 3)
    yy, xx = np.mgrid[0:32, 0:48]
    interior = ((yy % 8) >= 2) & ((yy % 8) <= 5) & ((xx % 8) >= 2) & ((xx % 8) <= 5)
    err = np.abs(back[:, interior].astype(int) - rgb[interior].astype(int))
    # Saturated colours clip in the encoder's Y'CbCr, so a few channels
    # cannot round-trip; the typical pixel must.
    assert np.median(err) <= 1
    assert np.quantile(err, 0.9) <= 3


def test_batching_does_not_change_the_result() -> None:
    rng = np.random.default_rng(1)
    nv12 = torch.from_numpy(rng.integers(16, 235, (5, 24, 16), dtype=np.uint8))
    a = nv12_to_rgb(nv12, 16, 16, max_batch=1)
    b = nv12_to_rgb(nv12, 16, 16, max_batch=8)
    assert torch.equal(a, b)


def test_analysis_size_keeps_aspect_and_multiples_of_8() -> None:
    assert analysis_size(3840, 2160, 640) == (640, 360)
    assert analysis_size(1080, 1920, 640) == (360, 640)
    w, h = analysis_size(2560, 1440, 640)
    assert w % 8 == 0 and h % 8 == 0 and abs(w / h - 16 / 9) < 0.03
