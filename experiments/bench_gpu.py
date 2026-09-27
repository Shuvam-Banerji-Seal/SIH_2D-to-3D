"""GPU throughput micro-benchmarks behind the paper's engineering claims.

* RAFT: stock torchvision vs the patched forward (no host syncs) vs CUDA-Graph
  replay, on real analysis frames, with the accuracy difference to stock.
* Decode: 4K VP9 through ffmpeg 4.4 (system) and 9.x (static) on NVDEC, and on
  the CPU, as source frames per second.
* nvJPEG: 4K JPEG encode latency on the GPU.

Every result carries the NVML reading of how busy the GPU already was, so a
number taken on a shared GPU says so.

    uv run python experiments/bench_gpu.py VIDEO paper/figures/bench_gpu.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402

import drone3d  # noqa: E402, F401  (first: undoes OpenMP thread binding before torch loads)
from drone3d.gpu.monitor import GpuMonitor  # noqa: E402
from drone3d.io.nvdec import (  # noqa: E402
    analysis_size,
    ffmpeg_bin,
    probe_stream,
    stream_analysis_chunks,
)
from drone3d.keyframes.flow import RaftFlow  # noqa: E402


def gpu_context() -> dict:
    with GpuMonitor(interval=0.2) as mon:
        time.sleep(1.0)
    s = mon.summary()
    return {
        "util_before_pct": s.util_mean,
        "foreign_processes": s.foreign_processes_max,
        "memory_used_gb": s.memory_peak_gb,
    }


def timed(fn, repeat: int = 3) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeat


def bench_raft(video: Path) -> dict:
    from torchvision.models import optical_flow as of

    info = probe_stream(video)
    size = analysis_size(info.width, info.height, 640)
    frames = []
    for _, rgb in stream_analysis_chunks(info, size, stride=2, chunk=65):
        frames.append(rgb)
        break
    frames = torch.cat(frames)[:65]
    a, b = frames[:-1], frames[1:]
    n = len(a)
    stock = of.raft_large(weights=of.Raft_Large_Weights.DEFAULT).cuda().eval()

    def run_stock(batch: int) -> torch.Tensor:
        outs = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, n, batch):
                outs.append(
                    stock(
                        RaftFlow._prep(a[s : s + batch]),
                        RaftFlow._prep(b[s : s + batch]),
                        num_flow_updates=12,
                    )[-1].float()
                )
        return torch.cat(outs)

    ref = run_stock(16)
    out = {"pairs": n, "size": list(size), "context": gpu_context()}
    out["stock_b16_flows_per_s"] = n / timed(lambda: run_stock(16))
    for graph in (False, True):
        fl = RaftFlow("raft_large", batch=32, cuda_graph=graph)
        res = fl(a, b)
        key = "graph_b32" if graph else "patched_b32"
        out[f"{key}_flows_per_s"] = n / timed(lambda fl=fl: fl(a, b))
        out[f"{key}_mean_abs_diff_px"] = float((res - ref).abs().mean())
    return out


def bench_decode(video: Path) -> dict:
    out = {"context": gpu_context()}
    info = probe_stream(video)
    variants = {
        "ffmpeg4_cuvid": ["/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-c:v", f"{info.codec}_cuvid", "-i", str(video), "-f", "null", "-"],
        "ffmpeg9_nvdec": [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video), "-f", "null", "-"],
        "ffmpeg9_cpu": [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-threads", "16", "-i", str(video), "-f", "null", "-"],
    }  # fmt: skip
    for name, cmd in variants.items():
        t = time.perf_counter()
        ok = subprocess.run(cmd, capture_output=True).returncode == 0
        dt = time.perf_counter() - t
        out[name] = {"seconds": dt, "source_fps": info.num_frames / dt if ok else None, "ok": ok}
    out["frames"] = info.num_frames
    out["resolution"] = [info.width, info.height]
    out["codec"] = info.codec
    return out


def bench_nvjpeg() -> dict:
    from torchvision.io import encode_jpeg

    x = torch.randint(0, 255, (3, 2160, 3840), dtype=torch.uint8, device="cuda")
    return {
        "ms_per_4k_frame": 1000 * timed(lambda: encode_jpeg(x, quality=95), repeat=20),
        "context": gpu_context(),
    }


def main() -> None:
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "raft": bench_raft(video),
        "decode": bench_decode(video),
        "nvjpeg": bench_nvjpeg(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
