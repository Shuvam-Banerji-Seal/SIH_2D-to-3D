"""Marigold v2 monocular depth (Pavlovic et al., SIGGRAPH Asia 2026), batched on the GPU.

Marigold v2 is a LoRA on the Qwen-Image-Edit-2509 diffusion transformer that
predicts affine-invariant *log* depth in a single rectified-flow step:

    VAE encode RGB -> one DiT step at t = 0.499 -> fine-tuned VAE decode -> mean(RGB)

The upstream code (``third_party/marigold-v2``) is imported from source and its
own registered loaders build the components, so the weights load exactly as in
its evaluation: a 4-bit NF4 base (what the LoRA was trained on), the rank-128
LoRA and the fine-tuned VAE decoder from ``trainables.safetensors``. Assets are
read from ``$DEPTH_ASSETS_DIR`` (default ``/store/huggingface/marigold-v2``).

The prediction is only defined up to ``log d = a * pred + b`` per image;
:mod:`drone3d.depth.align` recovers ``a, b`` from SfM points.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from drone3d.exceptions import BackendUnavailable
from drone3d.logging_utils import get_logger

__all__ = ["MarigoldDepth", "processing_size"]

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM = _REPO_ROOT / "third_party" / "marigold-v2"
_DEFAULT_ASSETS = Path("/store/huggingface/marigold-v2")
_EMBED_PREFIX = {
    "depth": "qwen_edit_2509_qwen_depth_realimg512",
    "normals": "qwen_edit_2509_qwen_normals_dummy512",
}
_LORA_TARGETS = [
    "img_in",
    "txt_in",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_add_out",
    "norm.linear",
    "a_to_out",
    "b_to_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "norm_out.linear",
    "proj_out",
]  # fmt: skip  (upstream evaluation/config/inference_depth.yaml)


def _quantized_cache(qwen: Path, cache_root: Path, quantization: str) -> Path:
    """Directory holding a pre-quantized copy of the base transformer, built on first use.

    The bf16 checkpoint is ~39 GB and is re-quantized on every load; the NF4
    copy is ~11 GB, so later loads read about a quarter of the bytes. The
    quantization settings are upstream's (``component_loader.py``), so the
    cached weights are the ones a fresh load would produce.
    """
    import torch
    from diffusers import BitsAndBytesConfig, QwenImageTransformer2DModel

    root = cache_root / f"Qwen-Image-Edit-2509-{quantization}"
    if (root / "transformer" / "config.json").is_file():
        return root
    log.info("building quantized transformer cache at %s (one-time)", root)
    if quantization == "4bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
    else:
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=["transformer_blocks.0.img_mod"]
        )
    model = QwenImageTransformer2DModel.from_pretrained(
        str(qwen),
        subfolder="transformer",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        quantization_config=qconfig,
    )
    tmp = root.with_name(root.name + ".tmp")
    model.save_pretrained(str(tmp / "transformer"))
    del model
    torch.cuda.empty_cache()
    tmp.rename(root)  # atomic publish: a crash mid-save leaves no half cache
    return root


def processing_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Aspect-preserving size with the given long side, both sides multiples of 16."""
    scale = long_side / max(width, height)
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


class MarigoldDepth:
    """Batched Marigold v2 inference.

    Args:
        checkpoint: sub-directory of ``Marigold-V2`` (``depth/Log-stage2`` is the
            paper model; ``normals`` predicts camera-space normals).
        quantization: ``4bit`` (matches training, ~12 GB), ``8bit`` or ``none`` (bf16).
        seed: seed of the VAE encoder's latent sample, reset per batch so
            results do not depend on batching.
    """

    def __init__(
        self,
        checkpoint: str = "depth/Log-stage2",
        *,
        assets_dir: str | Path | None = None,
        quantization: str = "4bit",
        device: str = "cuda",
        seed: int = 2025,
        cache: bool = True,
    ) -> None:
        assets = Path(assets_dir or os.environ.get("DEPTH_ASSETS_DIR", _DEFAULT_ASSETS))
        qwen = assets / "checkpoints" / "Qwen-Image-Edit-2509"
        marigold = assets / "checkpoints" / "Marigold-V2"
        trainables = marigold / checkpoint / "trainables.safetensors"
        for required in (qwen / "transformer", qwen / "vae", trainables):
            if not required.exists():
                raise BackendUnavailable(f"Marigold v2 asset missing: {required}")
        if not (_UPSTREAM / "marigoldv2").is_dir():
            raise BackendUnavailable(
                f"Marigold v2 source missing: {_UPSTREAM} (git submodule update)"
            )
        if str(_UPSTREAM) not in sys.path:
            sys.path.insert(0, str(_UPSTREAM))
        try:
            from marigoldv2.core.registry import REGISTRY
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise BackendUnavailable(
                f"Marigold v2 dependencies missing ({exc}); uv sync --extra depth"
            ) from exc

        modality = "normals" if checkpoint.startswith("normals") else "depth"
        transformer_root = qwen
        if cache and quantization in ("4bit", "8bit"):
            transformer_root = _quantized_cache(qwen, assets / "cache", quantization)
        cfg = OmegaConf.create(
            {
                "paths": {
                    "ckpt_qwen_image_edit": str(qwen),
                    "ckpt_qwen_image_edit_transformer": str(transformer_root),
                    "embed_dir": str(marigold / "qwen_text_embeddings"),
                },
                "optimization": {
                    "lora": {
                        "rank": 128,
                        "lora_alpha": 128,
                        "lora_dropout": 0.0,
                        "init_lora_weights": "gaussian",
                        "target_modules": _LORA_TARGETS,
                    },
                    "quantization": {"level": quantization},
                },
            }
        )
        REGISTRY["cfg"] = cfg
        pkg = "marigoldv2.experiments.20260316_qwen_depth"
        loader = importlib.import_module(f"{pkg}.component_loader")
        graph = importlib.import_module(f"{pkg}.network_graph")
        from marigoldv2.script.train.util import make_load_trainables_hook

        loader.LoadQwenImageEditVAE(name="VAE")()
        loader.LoadQwenImageEditTransformerFlexible(name="Diffuser")()
        make_load_trainables_hook(REGISTRY, None)([], str(marigold / checkpoint))
        for module in REGISTRY["network_components"].values():
            if isinstance(module, torch.nn.Module):
                module.requires_grad_(False)
                module.eval()
        self._encode = graph.QwenImageEncode(
            {"input_key": "rgb_norm", "output_key": "lat_encoding"}
        )
        self._step = graph.QwenImageEdit2509Step(
            {"prefix": _EMBED_PREFIX[modality], "predict_vel": True}
        )
        self._decode = graph.QwenImageDecode(
            {"input_key": "lat_encoding", "output_key": "pixel_pred"}
        )
        self.device = torch.device(device)
        self.seed = seed
        self.modality = modality
        self.checkpoint = checkpoint
        self.quantization = quantization
        log.info("Marigold v2 %s loaded (%s)", checkpoint, quantization)

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """``rgb`` uint8 ``[B, H, W, 3]`` -> prediction ``[B, C, h, w]`` at processing ``size``.

        Depth checkpoints return ``C = 1`` affine-invariant log depth (larger is
        farther); ``normals`` returns ``C = 3`` unit camera-space normals.
        """
        width, height = size
        x = rgb.to(self.device).permute(0, 3, 1, 2).float()
        x = F.interpolate(
            x, size=(height, width), mode="bicubic", antialias=True, align_corners=False
        )
        x = (x / 127.5 - 1.0).clamp(-1.0, 1.0)
        torch.manual_seed(self.seed)
        batch = {"rgb_norm": x, "out": {}}
        # Upstream runs inference under bf16 autocast (validate_steps.py): peft's
        # k-bit preparation keeps some layers in fp32.
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
            self._encode(batch)
            self._step(batch)
            self._decode(batch)
        pixel = batch["out"]["pixel_pred"].float()
        if self.modality == "normals":
            return F.normalize(pixel, dim=1)
        return pixel.mean(dim=1, keepdim=True)
