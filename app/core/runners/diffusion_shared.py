"""Stable Diffusion shared loader (full pipeline, no stubbing).
DRY/KISS: use hub snapshots by default; special-case segmind/tiny-sd which is
packaged as a Diffusers pipeline and loads best via its repo id.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import os
import time

import torch
from huggingface_hub import snapshot_download

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

log = logging.getLogger("app.runners.diffusion_shared")

# Cache SD pipelines per base model id
_SD_CACHE: Dict[str, Dict[str, Any]] = {}

# Models that should be loaded directly from the hub using their repo id,
# because they are published as full Diffusers pipelines rather than bare
# component folders (UNet/VAE/CLIP) that our local_dir logic expects.
DIRECT_SD_MODELS = {"segmind/tiny-sd"}

# Special tiny SD v1.5 variant that only ships safetensors weights, plus the
# base repo we can use for the full pipeline config.
TINY_SD15_MODEL_ID = "ehristoforu/stable-diffusion-v1-5-tiny"
BASE_SD15_REPO = "runwayml/stable-diffusion-v1-5"


def _download_model(model_id: str) -> str:
    """Resolve a SD model id to a local snapshot directory.

    We always go through snapshot_download; the HF cache ensures this only hits
    the network the first time for a given revision.
    """
    start = time.time()
    log.info(f"[sd] download start model={model_id} revision=latest")
    local_dir = snapshot_download(
        repo_id=model_id,
        max_workers=16,
        local_dir=None,
        local_dir_use_symlinks=True,
    )
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(fp)
            except Exception:
                pass
            file_count += 1
    dur = max(time.time() - start, 1e-6)
    mb = total_bytes / (1024**2)
    log.info(
        f"[sd] download done model={model_id} revision=latest files={file_count} "
        f"size_mb={mb:.2f} time_s={dur:.2f}"
    )
    return local_dir


def _has_model_index(local_dir: str) -> bool:
    """Return True if the downloaded snapshot looks like a Diffusers pipeline."""
    for root, _, files in os.walk(local_dir):
        if "model_index.json" in files:
            return True
    return False


def _is_tiny_sd15(model_id: str) -> bool:
    """Return True if this model id corresponds to the tiny SD v1.5 variant."""
    return model_id == TINY_SD15_MODEL_ID


def _find_tiny_sd15_unet(local_dir: str) -> Optional[str]:
    """Locate the tiny SD v1.5 UNet safetensors file in a local snapshot.

    The ehristoforu/stable-diffusion-v1-5-tiny repo only ships a couple of
    safetensors weights files (e.g. ``sd-v1-5-tiny.safetensors``) and no
    ``model_index.json``. We detect that layout and return the UNet weights
    path so we can bootstrap a pipeline from BASE_SD15_REPO.
    """
    candidates = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            if name.endswith(".safetensors") and "sd-v1-5-tiny" in name:
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    # Prefer a deterministic choice in case multiple matches exist.
    candidates.sort()
    return candidates[0]


def _apply_tiny_unet_weights(pipe_unet: torch.nn.Module, weights_path: str) -> None:
    """Load tiny UNet weights from safetensors into an existing UNet module."""
    try:
        from safetensors.torch import load_file as load_safetensors
    except Exception as exc:  # pragma: no cover - hard dependency path
        raise RuntimeError(
            "safetensors is required to load tiny SD v1.5 weights; "
            "please install safetensors."
        ) from exc

    log.info(f"[sd] loading tiny UNet weights from {weights_path}")
    state_dict = load_safetensors(weights_path, device="cpu")
    missing, unexpected = pipe_unet.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log.info(
            "[sd] tiny UNet load_state_dict non-strict: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def _bootstrap_tiny_sd15_pipeline(pipe_cls, unet_weights_path: str, device: Optional[torch.device]) -> Any:
    """Bootstrap a SD v1.5 tiny pipeline from the official SD 1.5 config.

    Important: we avoid relying on diffusers' internal component factory here
    because newer diffusers versions may pass extra kwargs such as
    ``offload_state_dict`` into CLIPTextModel and other submodules, which the
    transformers version bundled with this project does not accept. Instead we
    manually construct the pipeline from base SD 1.5 components and then swap
    in the tiny UNet weights.
    """

    from diffusers import DPMSolverMultistepScheduler

    dtype = torch.float16 if (device and device.type in ("cuda", "mps")) else None
    log.info(
        f"[sd] tiny-sd15 bootstrap start base={BASE_SD15_REPO} "
        f"dtype={'fp16' if dtype==torch.float16 else 'fp32'}"
    )

    # Download / resolve the base SD 1.5 snapshot once; rely on the HF cache
    # for repeated calls.
    base_dir = _download_model(BASE_SD15_REPO)

    # Load components explicitly, making sure we *never* pass experimental
    # kwargs like ``offload_state_dict`` into transformers models.
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        base_dir, subfolder="scheduler", local_files_only=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        base_dir, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_dir, subfolder="unet", torch_dtype=dtype, local_files_only=True
    )

    # Swap UNet weights with the tiny safetensors; this shrinks the model
    # without changing the surrounding architecture.
    _apply_tiny_unet_weights(unet, unet_weights_path)

    # Build the appropriate pipeline type.
    if pipe_cls is StableDiffusionImg2ImgPipeline:
        pipe = StableDiffusionImg2ImgPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
    else:
        # Default to the text-to-image pipeline.
        pipe = StableDiffusionPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

    if device:
        try:
            pipe.to(device)
        except Exception:
            # Device move failures should not be fatal for the whole server.
            pass
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    # safety_checker is already set to None when constructing the pipeline, but
    # keep this for defensive consistency with non-tiny init path.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return pipe


def _init_pipeline(pipe_cls, local_dir_or_id: str, device: Optional[torch.device], *, local: bool) -> Any:
    """Instantiate a Stable Diffusion pipeline.

    If local=True, ``local_dir_or_id`` is a filesystem path created by
    snapshot_download. Otherwise it is a hub repo id (for DIRECT_SD_MODELS
    or other models where we want Diffusers to resolve components itself).
    """
    start = time.time()
    dtype = torch.float16 if (device and device.type in ("cuda", "mps")) else None
    src = os.path.basename(local_dir_or_id) if local else local_dir_or_id
    log.info(
        f"[sd] init start model_src={src} "
        f"dtype={'fp16' if dtype==torch.float16 else 'fp32'} local={local}"
    )
    if local:
        pipe = pipe_cls.from_pretrained(local_dir_or_id, local_files_only=True, torch_dtype=dtype)
    else:
        pipe = pipe_cls.from_pretrained(local_dir_or_id, torch_dtype=dtype)
    if device:
        try:
            pipe.to(device)
        except Exception:
            pass
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    dur = time.time() - start
    try:
        params = sum(p.numel() for p in pipe.unet.parameters())
        mem_mb = (params * (2 if dtype == torch.float16 else 4)) / (1024**2)
    except Exception:
        params = 0
        mem_mb = 0
    log.info(f"[sd] init done params={params} est_mem_mb={mem_mb:.2f} time_s={dur:.2f}")
    return pipe


def get_or_create_sd_pipeline(model_id: str, device: Optional[torch.device], mode: str) -> Dict[str, Any]:
    """Return cached SD pipelines for a given model.

    Cache key is the base model_id; per-process cache avoids re-init of heavy
    pipelines. The HF snapshot cache handles network avoidance.
    """
    entry = _SD_CACHE.get(model_id)
    if entry is None:
        entry = {"pipe_text": None, "pipe_img2img": None}
        _SD_CACHE[model_id] = entry
    if mode not in {"text", "img2img"}:
        raise ValueError("mode must be 'text' or 'img2img'")

    is_tiny = _is_tiny_sd15(model_id)
    use_direct = model_id in DIRECT_SD_MODELS

    # Lazily resolve tiny SD15 UNet path if needed.
    tiny_unet_path: Optional[str] = None
    if is_tiny:
        local = _download_model(model_id)
        tiny_unet_path = _find_tiny_sd15_unet(local)
        if not tiny_unet_path:
            log.error(
                "[sd] tiny sd15 layout not recognized in %s; falling back to "
                "standard hub loading which is expected to fail.",
                local,
            )

    if mode == "text" and entry["pipe_text"] is None:
        if is_tiny and tiny_unet_path:
            entry["pipe_text"] = _bootstrap_tiny_sd15_pipeline(
                StableDiffusionPipeline, tiny_unet_path, device
            )
        elif use_direct:
            entry["pipe_text"] = _init_pipeline(
                StableDiffusionPipeline, model_id, device, local=False
            )
        else:
            local = _download_model(model_id)
            if _has_model_index(local):
                entry["pipe_text"] = _init_pipeline(
                    StableDiffusionPipeline, local, device, local=True
                )
            else:
                entry["pipe_text"] = _init_pipeline(
                    StableDiffusionPipeline, model_id, device, local=False
                )

    if mode == "img2img" and entry["pipe_img2img"] is None:
        if is_tiny and tiny_unet_path:
            entry["pipe_img2img"] = _bootstrap_tiny_sd15_pipeline(
                StableDiffusionImg2ImgPipeline, tiny_unet_path, device
            )
        elif use_direct:
            entry["pipe_img2img"] = _init_pipeline(
                StableDiffusionImg2ImgPipeline, model_id, device, local=False
            )
        else:
            local = _download_model(model_id)
            if _has_model_index(local):
                entry["pipe_img2img"] = _init_pipeline(
                    StableDiffusionImg2ImgPipeline, local, device, local=True
                )
            else:
                entry["pipe_img2img"] = _init_pipeline(
                    StableDiffusionImg2ImgPipeline, model_id, device, local=False
                )

    return entry


__all__ = ["get_or_create_sd_pipeline"]
