"""One decode, two resolutions: the sidecar frames match the analysis frames, frame for frame."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("hwaccel", [False, True])
def test_sidecar_frames_are_the_analysis_frames_at_another_size(tmp_path: Path, hwaccel: bool) -> None:
    from drone3d.io.nvdec import (
        ffmpeg_bin,
        probe_stream,
        stream_analysis_chunks,
        write_sidecar_jpegs,
    )

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=10", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    info = probe_stream(clip)
    plain = list(stream_analysis_chunks(info, (160, 90), stride=2, chunk=8, hwaccel=hwaccel))
    stacked = list(stream_analysis_chunks(info, (160, 90), stride=2, chunk=8, hwaccel=hwaccel, sidecar=(320, 180)))
    idx_plain = np.concatenate([c[0] for c in plain])
    idx_stacked = np.concatenate([c[0] for c in stacked])
    np.testing.assert_array_equal(idx_plain, idx_stacked)
    an = torch.cat([c[1] for c in stacked]).float()
    side = torch.cat([c[2] for c in stacked]).float()
    assert an.shape[1:3] == (90, 160) and side.shape[1:3] == (180, 320)
    # the same analysis frames as the plain stream, and the sidecar is the same picture
    torch.testing.assert_close(an, torch.cat([c[1] for c in plain]).float(), atol=2.0, rtol=0)
    down = F.interpolate(side.permute(0, 3, 1, 2), size=(90, 160), mode="area").permute(0, 2, 3, 1)
    assert float((down - an).abs().mean()) < 6.0

    out = tmp_path / "cand"
    frames = list(write_sidecar_jpegs(iter(stacked), out))
    assert sum(len(f[0]) for f in frames) == len(idx_stacked)
    assert sorted(p.name for p in out.glob("*.jpg")) == [f"f_{i:06d}.jpg" for i in idx_stacked]
