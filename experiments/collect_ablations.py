"""Collect the ablation numbers the paper reports into paper/figures/ablations.json.

Each row is read from the file that produced it (an SfM log, a COLMAP model, a
training metrics.json); nothing is typed in. Missing inputs are skipped.

    uv run python experiments/collect_ablations.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "paper" / "figures" / "ablations.json"


def sfm_row(label: str, sparse: Path, frames: int, note: str = "") -> dict | None:
    import pycolmap

    if not sparse.is_dir():
        return None
    sizes = []
    for model in sorted(p for p in sparse.iterdir() if (p / "images.bin").is_file()):
        rec = pycolmap.Reconstruction(str(model))
        sizes.append(sum(1 for im in rec.images.values() if im.has_pose))
    if not sizes:
        return None
    return {"label": label, "frames": frames, "registered": sum(sizes), "models": len(sizes), "largest": max(sizes), "note": note}


def reproj(log: Path) -> float | None:
    m = re.search(r"Reprojection error: mean ([\d.]+) px", log.read_text()) if log.is_file() else None
    return float(m.group(1)) if m else None


def camera_row(model_name: str, sparse: Path, log: Path) -> dict | None:
    import pycolmap

    if not sparse.is_dir():
        return None
    focals, points = [], 0
    for model in sorted(p for p in sparse.iterdir() if (p / "images.bin").is_file()):
        rec = pycolmap.Reconstruction(str(model))
        cam = next(iter(rec.cameras.values()))
        params = list(cam.params)
        focals.append(f"{params[0]:.0f}/{params[1]:.0f}" if str(cam.model).endswith("OPENCV") else f"{params[0]:.0f}")
        points += len(rec.points3D)
    return {"model": model_name, "focals": ", ".join(focals), "reproj": reproj(log), "points": points}


def splat_row(run: str, label: str, run_dir: Path) -> dict | None:
    metrics = run_dir / "metrics.json"
    if not metrics.is_file():
        return None
    m = json.loads(metrics.read_text())
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").is_file() else {}
    log = run_dir.parent / f"{run_dir.name}.log"
    splats = [int(x) for x in re.findall(r"splats (\d+)", log.read_text())] if log.is_file() else []
    return {
        "run": run,
        "label": label,
        "num_splats": splats[-1] if splats else None,
        "psnr": m.get("avg_psnr"),
        "ssim": m.get("avg_ssim"),
        "cc_psnr": m.get("avg_cc_psnr"),
        "lpips": m.get("avg_lpips"),
        "seconds": m.get("training_time"),
        "eval_views": m.get("num_eval_images"),
        "depth_weight": cfg.get("depth_supervision_weight"),
    }


def main() -> None:
    dev = ROOT / "outputs" / "dev_jal"
    canon = ROOT / "outputs" / "jal_mahal"
    rows = [
        sfm_row("1 fps (uniform)", dev / "baseline_1fps" / "sparse", 55, "no letterbox crop"),
        sfm_row("uniform, same budget", dev / "baseline_uniform241" / "sparse", 241, "no letterbox crop"),
        sfm_row("overlap band (dev)", dev / "dataset_radial" / "sparse", 241, "no letterbox crop"),
        sfm_row("overlap band (canonical)", canon / "dataset" / "sparse", 213, "cropped, fades trimmed"),
    ]
    camera = [
        camera_row("OpenCV (fx, fy free)", dev / "dataset" / "sparse", dev / "sfm.log"),
        camera_row("radial (one focal)", dev / "dataset_radial" / "sparse", dev / "sfm_radial.log"),
    ]
    splat = [
        splat_row("jal_mahal", "model_0, RGB only", canon / "ablation" / "model_0_rgb_only"),
        splat_row("jal_mahal", "model_0, depth prior (v1 mask)", canon / "splats" / "model_0"),
        splat_row("jal_mahal", "model_0, depth prior (v2 mask)", canon / "ablation" / "model_0_depth_v2"),
    ]
    payload = {
        "sampling": [r for r in rows if r],
        "camera": [r for r in camera if r],
        "splat": [r for r in splat if r],
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
