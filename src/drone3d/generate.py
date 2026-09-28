"""A generated object beside the measured model: TRELLIS.2 on the keyframe that shows the building whole.

The measured model (flow SfM, fused depth) is only as complete as the flight: the far side of a building
the drone never circled is missing, a short shot leaves gaps. TRELLIS.2 (microsoft/TRELLIS.2-4B) generates a
complete, textured object from one image. It is *not* a measurement -- the sides the keyframe does not
show are invented and the scale is arbitrary -- so it is written apart (``export/generated/``), listed in
the model catalog as generated, and never enters the deliverables or the metrics.

It runs in TRELLIS.2's own environment (``tools/setup_trellis2.sh``: torch 2.7, its CUDA extensions) as a
subprocess of ``tools/trellis2_generate.py``: ~2.5 min to load the weights, ~80 s to generate at 1024^3 and
~80 s to bake the GLB on the A100, 8 GB of GPU memory at most.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "third_party" / "TRELLIS.2" / ".venv" / "bin" / "python"
SCRIPT = ROOT / "tools" / "trellis2_generate.py"
NOTE = "generated from one keyframe by TRELLIS.2, not measured: unseen sides are invented and the scale is arbitrary"
LICENSES = {"microsoft/TRELLIS.2-4B": "MIT", "briaai/RMBG-2.0": "bria-rmbg-2.0 (non-commercial)",
            "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3-license"}  # fmt: skip


def available() -> bool:
    """TRELLIS.2's environment and the wrapper script are installed."""
    return PYTHON.is_file() and SCRIPT.is_file()


def subject_keyframe(cams: np.ndarray, axes: np.ndarray, *, cone_deg: float = 12.0) -> int:
    """Index of the keyframe that shows the subject whole.

    The subject is what the optical axes converge on (:func:`drone3d.export.stage._subject`); of the
    keyframes looking at it within ``cone_deg``, the farthest from it frames it whole -- on Colosseum's
    merged model, a view of the entire amphitheatre, where the nearest showed one arch. Axes that hardly
    converge (a survey, a fly-by) give the middle keyframe.
    """
    from drone3d.export.stage import _subject

    mid = len(cams) // 2
    sub = _subject(cams, axes)
    if sub is None:
        return mid
    rel = sub[0] - cams
    dist = np.linalg.norm(rel, axis=1)
    looking = np.einsum("ij,ij->i", rel, axes) / np.maximum(dist, 1e-12) > np.cos(np.radians(cone_deg))
    if not looking.any():
        return mid
    return int(np.flatnonzero(looking)[np.argmax(dist[looking])])


def pick_keyframe(run_dir: Path) -> tuple[Path, dict]:
    """The largest model's subject keyframe -> ``(image path, info)``."""
    import pycolmap

    sfm = json.loads((run_dir / "sfm" / "result.json").read_text())
    models = sorted(sfm.get("models", []), key=lambda m: -(m.get("images") or 0))
    if not models:
        raise ValueError("no SfM model: run the pipeline first")
    rec = pycolmap.Reconstruction(models[0]["path"])
    posed = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if im.has_pose]
    cams = np.array([im.projection_center() for im in posed])
    axes = np.array([im.cam_from_world().rotation.matrix()[2] for im in posed])
    k = subject_keyframe(cams, axes)
    return run_dir / "dataset" / "images" / posed[k].name, {"model": Path(models[0]["path"]).name, "keyframe": posed[k].name}


def out_dir(run_dir: Path) -> Path:
    return run_dir / "export" / "generated"


def status(run_dir: Path) -> dict | None:
    f = out_dir(run_dir) / "result.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def generate_object(run_dir: Path, *, image: Path | None = None, resolution: str = "1024", seed: int = 0,
                    timeout_s: float = 1800.0) -> dict:  # fmt: skip
    """Generate ``export/generated/object.glb`` for a finished run -> the result record (also result.json)."""
    if not available():
        raise RuntimeError(f"TRELLIS.2 is not installed ({PYTHON}); run tools/setup_trellis2.sh")
    out = out_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"keyframe": None}
    if image is None:
        image, info = pick_keyframe(run_dir)
    record = {"status": "running", "started": time.time(), "pid": os.getpid(), "image": str(image), **info,
              "resolution": resolution, "seed": seed, "generator": "microsoft/TRELLIS.2-4B", "note": NOTE,
              "licenses": LICENSES}  # fmt: skip
    (out / "result.json").write_text(json.dumps(record, indent=1))
    glb = out / "object.glb"
    # the shell's OpenMP binding pins a child to one core (see drone3d.__init__)
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([str(PYTHON), str(SCRIPT), str(glb), str(image), "--res", resolution, "--seed", str(seed)],
                              capture_output=True, text=True, env=env, timeout=timeout_s, cwd=ROOT)  # fmt: skip
        ok = proc.returncode == 0 and glb.is_file()
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    except subprocess.TimeoutExpired:
        ok, tail = False, [f"timed out after {timeout_s:.0f} s"]
    record.update(status="ok" if ok else "failed", seconds=round(time.perf_counter() - t0, 1), finished=time.time(),
                  glb="object.glb" if ok else None, input="object.input.png" if (out / "object.input.png").is_file() else None,
                  log=tail)  # fmt: skip
    (out / "result.json").write_text(json.dumps(record, indent=1))
    if not ok:
        log.warning("generate %s failed: %s", run_dir.name, " | ".join(tail[-3:]))
    return record
