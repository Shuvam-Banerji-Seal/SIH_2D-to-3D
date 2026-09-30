"""What clears the haze out of the 360-degree splats?

The highrise's splats (``drone3d.complete.splats_360``) carry a veil of sky-coloured splats in front of the
tower: the trainer's background is a constant colour, so every sky pixel of a keyframe has to be painted
by splats, and it paints them where it can -- near the subject. Variants:

  * ``colour``: as ``splats_360`` trains (sky painted; masks ignore what nothing supervises);
  * ``skybox``: spirula's ``--background-mode sh``, a learned skybox for the sky;
  * ``empty``: ``--apply-loss-for-mask``, the sky trained as empty space -- the keyframes' sky (Depth
    Anything V2, the dense stage's rule, eroded 3 px at 640 px as ``drone3d.complete`` carves with it) and,
    in the views of unflown headings, all but the complete model;
  * ``empty_full``: the same with the whole sky (the erosion undone): the band the erosion left was trained
    as sky colour, a light halo round the tower's crown;
  * ``empty_grown``: the whole sky grown up to the skyline (``drone3d.complete.sky_to_edges``);
  * ``empty_low``: that, without the views above the flown headings (``splats_360`` supervises only the
    subject there; with every pixel either kept or empty, they taught the broken measured crown);
  * ``sky_background``: as ``colour``, the trainer's background the photographs' median sky colour -- the sky
    explained by no splat at all rather than painted; ``sky_background_fs`` with spirula's mild floater
    suppression (depth and colour distortion losses, a cap on on-screen splat size).

The web viewer draws no skybox, so every metric renders the splats alone, on black:

  * ``ground_psnr``: PSNR over the held-out keyframes' pixels 9 px (at 640 px) or more from the sky -- the
    photographs, reproduced (the band at the sky's edge, black in an empty sky, blue in the photographs,
    cost 6 dB alone);
  * ``sky_alpha``: the splats' mean opacity over those keyframes' sky -- the veil seen from the flight;
  * ``virtual_psnr``: PSNR over the held-out views of unflown headings, on the complete model's pixels;
  * ``virtual_sky_alpha``: opacity over those views' sky (nothing modelled, above the horizon).

    uv run python experiments/splat_sky.py outputs/new_highrise_orbit [--iterations 7000] [--out DIR]

Trains on copies of ``export/complete/splat_data`` (``splats_360(train=False)`` writes it); a variant
already trained under ``--out`` is evaluated again, not retrained.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import drone3d  # noqa: F401  (unpins OpenMP threads)

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")  # gsplat compiles its kernels on first use

VARIANTS = {  # name -> (dataset, trainer flags)
    "colour": ("splat_data", {}),
    "skybox": ("splat_data", {"background_mode": "sh"}),
    "empty": ("splat_data_empty", {"apply_loss_for_mask": 1}),
    "empty_full": ("splat_data_empty_full", {"apply_loss_for_mask": 1}),
    "empty_grown": ("splat_data_empty_grown", {"apply_loss_for_mask": 1}),
    "empty_low": ("splat_data_empty_low", {"apply_loss_for_mask": 1}),
    "sky_background": ("splat_data", {"background_color": "sky"}),  # the photographs' median sky colour
    "sky_background_fs": ("splat_data", {"background_color": "sky", "floater_suppression": "mild"}),
}


def _resized(b: np.ndarray, w: int, h: int) -> np.ndarray:
    """A boolean mask at w x h (nearest)."""
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(b).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127


class _V:  # the minimum drone3d.complete._sky_masks reads
    def __init__(self, img):  # type: ignore[no-untyped-def]
        self.image = img


def views_of(run: Path, data: Path, scene: Path) -> dict:
    """Per image of the dataset: what the evaluation and the ``empty`` masks need.

    Keyframes: ``sky`` (bool, full resolution; eroded), ``sky_full`` (the erosion undone), ``sky_grown``,
    ``ground`` (9 px or more from the sky). Views of unflown headings: ``hit`` (the complete model's
    pixels, re-rendered from ``scene`` -- the complete model the views were rendered from) and ``up`` (rows
    above the horizon).
    """
    import pycolmap
    import torch
    from scipy import ndimage
    from torchvision.io import read_file

    from drone3d.complete import _sky_masks, ray_elevation, sky_to_edges
    from drone3d.export.render_mesh import load_parts, render_parts
    from drone3d.gpu.nvjpeg import decode_jpeg

    sfm = json.loads((run / "sfm" / "result.json").read_text())
    name = Path(sfm["models"][0]["path"]).name
    frame = json.loads((run / "export" / f"model_{name}" / "frame.json").read_text())
    fs, fr, ft = float(frame["scale"]), np.asarray(frame["rotation"]), np.asarray(frame["translation"])
    rec = pycolmap.Reconstruction(str(data / "sparse" / "0"))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    out: dict = {}
    keys = [im for im in ims if not im.name.startswith("virtual/")]
    for k in range(0, len(keys), 8):  # Depth Anything in batches of views
        batch = keys[k : k + 8]
        imgs = [decode_jpeg(read_file(str(data / "images" / im.name)), device="cuda").permute(1, 2, 0).contiguous() for im in batch]
        for im, img, sky in zip(batch, imgs, _sky_masks([_V(i) for i in imgs]), strict=True):
            h, w = img.shape[:2]
            out[im.name] = {"sky": _resized(sky, w, h), "sky_full": _resized(ndimage.binary_dilation(sky, iterations=3), w, h),
                            "sky_grown": sky_to_edges(img.cpu().numpy(), sky, ray_elevation(
                                rec.cameras[im.camera_id], np.asarray(im.cam_from_world().rotation.matrix()) @ fr.T, (w, h))),
                            "ground": _resized(~ndimage.binary_dilation(sky, iterations=9), w, h)}  # fmt: skip
        del imgs
        torch.cuda.empty_cache()
    parts = load_parts(scene)
    for im in ims:
        if not im.name.startswith("virtual/"):
            continue
        cam = rec.cameras[im.camera_id]
        w, h, fpx = int(cam.width), int(cam.height), float(cam.focal_length_x)
        pose = im.cam_from_world()
        rot = np.asarray(pose.rotation.matrix()) @ fr.T  # world -> camera, in the export frame
        c = fs * fr @ im.projection_center() + ft
        _, part = render_parts(parts, fpx, rot, c, (w, h))
        ys = np.arange(h) + 0.5
        up = (rot.T @ np.stack([np.zeros(h), (ys - h / 2) / fpx, np.ones(h)]))[2] > 0
        out[im.name] = {"hit": part >= 0, "up": np.broadcast_to(up[:, None], (h, w))}
    return out


def empty_dataset(data: Path, dst: Path, views: dict, sky: str = "sky", drop_above: bool = False) -> None:
    """``data`` with masks for ``--apply-loss-for-mask``: kept = what is there, masked out = empty space.
    ``drop_above``: without the views whose ``splats_360`` mask leaves part of the model out (above the flight)."""
    import tempfile

    import pycolmap
    from PIL import Image

    if dst.exists():
        return
    dst.mkdir(parents=True)
    (dst / "images").symlink_to((data / "images").resolve())
    drop = set()
    if drop_above:
        for name, v in views.items():
            if "hit" in v and (v["hit"] & ~(np.asarray(Image.open(data / "masks" / Path(name).with_suffix(".png"))) > 127)).any():
                drop.add(name)
    if not drop:
        shutil.copytree(data / "sparse", dst / "sparse")
    else:  # the model without them, through COLMAP's text files
        rec = pycolmap.Reconstruction(str(data / "sparse" / "0"))
        ids = {i for i, im in rec.images.items() if im.name in drop}
        with tempfile.TemporaryDirectory() as tmp:
            rec.write_text(tmp)
            lines = (Path(tmp) / "images.txt").read_text().splitlines()
            body = [ln for ln in lines if not ln.startswith("#")]
            kept = []
            for head, pts in zip(body[0::2], body[1::2], strict=True):
                if int(head.split()[0]) not in ids:
                    kept += [head, pts]
            (Path(tmp) / "images.txt").write_text("\n".join(kept) + "\n")
            frames = [ln for ln in (Path(tmp) / "frames.txt").read_text().splitlines()
                      if ln.startswith("#") or int(ln.split()[-1]) not in ids]  # fmt: skip
            (Path(tmp) / "frames.txt").write_text("\n".join(frames) + "\n")
            out = dst / "sparse" / "0"
            out.mkdir(parents=True)
            pycolmap.Reconstruction(tmp).write_binary(str(out))
        print(f"{dst.name}: {len(drop)} views above the flight left out", flush=True)
    for name, v in views.items():
        if name in drop:
            continue
        keep = ~v[sky] if sky in v else v["hit"]
        path = dst / "masks" / Path(name).with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(keep.astype(np.uint8) * 255).save(path)


def evaluate(data: Path, trained: Path, views: dict) -> dict:
    """The metrics of the module docstring for one trained model."""
    import gsplat
    import pycolmap
    import torch
    from PIL import Image

    from drone3d.splat.render import load_splat_ply

    rec = pycolmap.Reconstruction(str(data / "sparse" / "0"))
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    held = [im for i, im in enumerate(ims) if i % 8 == 0]  # spirula's --eval-mode interval --eval-interval 8
    ply = sorted(trained.rglob("splat*.ply"), key=lambda p: p.stat().st_mtime)[-1]
    tf = json.loads((trained / "scene_transform.json").read_text())["train_from_world"]
    s, r, t = float(tf["scale"]), np.array(tf["rotation"]["matrix_3x3"]), np.array(tf["translation"])
    spl = load_splat_ply(ply)
    m: dict[str, list] = {"ground": [], "sky": [], "virtual": [], "virtual_sky": []}
    psnr = lambda e: 10 * np.log10(1 / max(float(e), 1e-10))  # noqa: E731
    for im in held:
        cam = rec.cameras[im.camera_id]
        w, h = cam.width // 2, cam.height // 2
        fx, cx, cy = cam.focal_length_x / 2, cam.principal_point_x / 2, cam.principal_point_y / 2
        k = torch.tensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")
        pose = im.cam_from_world()
        rot, c = np.asarray(pose.rotation.matrix()), im.projection_center()
        vm = np.eye(4)
        vm[:3, :3] = rot @ r.T
        vm[:3, 3] = s * (-rot @ c) - rot @ r.T @ t
        with torch.no_grad():
            img, acc, _ = gsplat.rasterization(
                spl["means"], spl["quats"], spl["scales"], spl["opacities"], spl["sh"],
                torch.tensor(vm[None], dtype=torch.float32, device="cuda"), k[None], w, h,
                sh_degree=spl["sh_degree"], render_mode="RGB",
            )  # fmt: skip
        ren, a = img[0].clamp(0, 1).cpu().numpy(), acc[0, ..., 0].cpu().numpy()
        gt = np.asarray(Image.open(data / "images" / im.name).convert("RGB").resize((w, h), Image.BICUBIC)) / 255.0
        v = views[im.name]
        if "sky" in v:
            sky = _resized(v["sky"], w, h)
            m["ground"].append(psnr(((ren - gt) ** 2)[_resized(v["ground"], w, h)].mean()))
            if sky.any():
                m["sky"].append(float(a[sky].mean()))
        else:
            hit, open_sky = _resized(v["hit"], w, h), _resized(~v["hit"] & v["up"], w, h)
            m["virtual"].append(psnr(((ren - gt) ** 2)[hit].mean()))
            if open_sky.any():
                m["virtual_sky"].append(float(a[open_sky].mean()))
    mean = lambda x, nd: round(float(np.mean(x)), nd) if x else None  # noqa: E731
    return {
        "ground_psnr": mean(m["ground"], 2), "sky_alpha": mean(m["sky"], 3),
        "virtual_psnr": mean(m["virtual"], 2), "virtual_sky_alpha": mean(m["virtual_sky"], 3),
        "held_keyframes": len(m["ground"]), "held_virtual": len(m["virtual"]), "splats": int(len(spl["means"])),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--scene", type=Path, default=None, help="the complete scene the views were rendered from")
    a = ap.parse_args()
    from PIL import Image

    from drone3d.splat.spirula import run_train

    work = a.out or a.run / "experiments" / "splat_sky"
    data = work / "splat_data"
    if not data.exists():
        shutil.copytree(a.run / "export" / "complete" / "splat_data", data, symlinks=True)
    views = views_of(a.run, data, a.scene or a.run / "export" / "complete" / "scene.glb")
    empty_dataset(data, work / "splat_data_empty", views)
    empty_dataset(data, work / "splat_data_empty_full", views, sky="sky_full")
    empty_dataset(data, work / "splat_data_empty_grown", views, sky="sky_grown")
    empty_dataset(data, work / "splat_data_empty_low", views, sky="sky_grown", drop_above=True)
    results = json.loads((work / "results.json").read_text()) if (work / "results.json").is_file() else {}
    for name in a.variants:
        ds, flags = VARIANTS[name]
        if flags.get("background_color") == "sky":
            sky_px = np.concatenate([np.asarray(Image.open(data / "images" / n).convert("RGB"))[v["sky"]]
                                     for n, v in views.items() if "sky" in v])  # fmt: skip
            flags = {**flags, "background_color": tuple(round(float(x) / 255, 4) for x in np.median(sky_px, axis=0))}
        out = work / name
        if not any(out.rglob("splat*.ply")):
            run_train(work / ds, out, preset="3dgs", recon_dir="sparse/0", iterations=a.iterations, quality="medium",
                      depth_weight=0.0, eval_interval=8,
                      flags={"train_resolution_divisor": 2, "cache_images": "disk", "load_depths": 0, **flags})  # fmt: skip
        res = evaluate(work / ds, out, views)
        metrics = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").is_file() else {}
        res.update(trainer_psnr=metrics.get("avg_psnr"), training_s=metrics.get("training_time"), flags=flags)
        results[name] = res
        print(name, json.dumps(res), flush=True)
        (work / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
