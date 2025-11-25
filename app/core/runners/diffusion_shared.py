"""Stable Diffusion shared loader (full pipeline, no stubbing).

This loader dynamically detects pipeline architectures from model_index.json
and constructs the correct diffusers pipeline *without* passing unsupported
arguments (e.g. offload_state_dict) into transformers components.

Supports:
- StableDiffusionPipeline (text-to-image)
- StableDiffusionImg2ImgPipeline (img2img)
- TextToVideoSDPipeline (text-to-video)
- tiny SD15 bootstrap logic (unchanged)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Type
import logging
import os
import json
import time

import torch
from huggingface_hub import snapshot_download

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

log = logging.getLogger("app.runners.diffusion_shared")

_SD_CACHE: Dict[str, Dict[str, Any]] = {}
_SCHEDULER_CACHE: Dict[Tuple[str, str], Any] = {}

DIRECT_SD_MODELS = {"segmind/tiny-sd"}

TINY_SD15_MODEL_ID = "ehristoforu/stable-diffusion-v1-5-tiny"
BASE_SD15_REPO = "runwayml/stable-diffusion-v1-5"

# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #

def _download_model(model_id: str) -> str:
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


def _load_model_index(local_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(local_dir, "model_index.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _has_model_index(local_dir: str) -> bool:
    return os.path.exists(os.path.join(local_dir, "model_index.json"))


def _get_or_create_scheduler(local_dir: str, scheduler_cls: Type[Any]) -> Any:
    """Return a cached diffusion scheduler for a given local directory.

    Caches by (local_dir, scheduler_cls.__name__) to avoid repeated
    from_pretrained() calls across multiple pipelines and invocations.
    """
    key = (os.path.abspath(local_dir), scheduler_cls.__name__)
    cached = _SCHEDULER_CACHE.get(key)
    if cached is not None:
        return cached
    scheduler = scheduler_cls.from_pretrained(local_dir, subfolder="scheduler", local_files_only=True)
    _SCHEDULER_CACHE[key] = scheduler
    return scheduler

# --------------------------------------------------------------------------- #
# Tiny SD15 helper
# --------------------------------------------------------------------------- #

def _is_tiny_sd15(model_id: str) -> bool:
    return model_id == TINY_SD15_MODEL_ID


def _find_tiny_sd15_unet(local_dir: str) -> Optional[str]:
    candidates = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            if name.endswith(".safetensors") and "sd-v1-5-tiny" in name:
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def _apply_tiny_unet_weights(pipe_unet: torch.nn.Module, weights_path: str) -> None:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors is required to load tiny SD v1.5 weights."
        ) from exc

    log.info(f"[sd] loading tiny UNet weights from {weights_path}")
    state_dict = load_file(weights_path, device="cpu")
    missing, unexpected = pipe_unet.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log.info(
            "[sd] tiny UNet load_state_dict non-strict: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def _bootstrap_tiny_sd15_pipeline(pipe_cls, unet_weights_path: str, device):
    from diffusers import DPMSolverMultistepScheduler

    dtype = torch.float16 if (device and device.type in ("cuda", "mps")) else None
    log.info(f"[sd] tiny-sd15 bootstrap start base={BASE_SD15_REPO}")

    base_dir = _download_model(BASE_SD15_REPO)

    scheduler = _get_or_create_scheduler(base_dir, DPMSolverMultistepScheduler)
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

    _apply_tiny_unet_weights(unet, unet_weights_path)

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
        try: pipe.to(device)
        except Exception: pass

    if hasattr(pipe, "enable_attention_slicing"):
        try: pipe.enable_attention_slicing()
        except Exception: pass

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    return pipe


# --------------------------------------------------------------------------- #
# Text-to-video bootstrap (generic)
# --------------------------------------------------------------------------- #

def _bootstrap_text_to_video_pipeline(local_dir: str, device):
    """Construct a TextToVideoSDPipeline manually from a local snapshot."""
    from diffusers import TextToVideoSDPipeline, UNet3DConditionModel
    from diffusers import DPMSolverMultistepScheduler

    model_index = _load_model_index(local_dir)
    if not model_index:
        raise RuntimeError("model_index.json missing for text-to-video pipeline")

    dtype = torch.float16 if (device and device.type in ("cuda", "mps")) else None

    scheduler = _get_or_create_scheduler(local_dir, DPMSolverMultistepScheduler)
    tokenizer = CLIPTokenizer.from_pretrained(
        local_dir, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        local_dir, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        local_dir, subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    unet = UNet3DConditionModel.from_pretrained(
        local_dir, subfolder="unet", torch_dtype=dtype, local_files_only=True
    )

    pipe = TextToVideoSDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if device:
        try: pipe.to(device)
        except Exception: pass

    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass

    return pipe


# --------------------------------------------------------------------------- #
# Generic bootstrap for model_index.json
# --------------------------------------------------------------------------- #

def _bootstrap_from_model_index(local_dir: str, device: Optional[torch.device]):
    """Load pipeline based on its model_index.json _class_name."""
    model_index = _load_model_index(local_dir)
    if not model_index:
        return None

    cls = model_index.get("_class_name", "")
    if cls == "TextToVideoSDPipeline":
        return _bootstrap_text_to_video_pipeline(local_dir, device)

    # fall back and let other logic handle it
    return None


# --------------------------------------------------------------------------- #
# Stable diffusion loaders
# --------------------------------------------------------------------------- #

def _init_pipeline(pipe_cls, local_dir_or_id: str, device, *, local: bool):
    start = time.time()
    dtype = torch.float16 if (device and device.type in ("cuda", "mps")) else None
    src = os.path.basename(local_dir_or_id) if local else local_dir_or_id
    log.info(f"[sd] init start src={src}")

    if local:
        pipe = pipe_cls.from_pretrained(
            local_dir_or_id,
            local_files_only=True,
            torch_dtype=dtype,
        )
    else:
        pipe = pipe_cls.from_pretrained(
            local_dir_or_id,
            torch_dtype=dtype,
        )

    if device:
        try: pipe.to(device)
        except Exception: pass

    if hasattr(pipe, "enable_attention_slicing"):
        try: pipe.enable_attention_slicing()
        except Exception: pass

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    dur = time.time() - start
    try:
        params = sum(p.numel() for p in pipe.unet.parameters())
        mem_mb = params * (2 if dtype == torch.float16 else 4) / (1024**2)
    except Exception:
        params = 0
        mem_mb = 0

    log.info(f"[sd] init done params={params} est_mem_mb={mem_mb:.2f} time_s={dur:.2f}")
    return pipe


def get_or_create_sd_pipeline(model_id: str, device: Optional[torch.device], mode: str) -> Dict[str, Any]:
    if mode not in {"text", "img2img"}:
        raise ValueError("mode must be 'text' or 'img2img'")

    entry = _SD_CACHE.get(model_id)
    if entry is None:
        entry = {"pipe_text": None, "pipe_img2img": None}
        _SD_CACHE[model_id] = entry

    is_tiny = _is_tiny_sd15(model_id)
    use_direct = model_id in DIRECT_SD_MODELS

    # tiny SD15 prep
    tiny_unet_path = None
    if is_tiny:
        local = _download_model(model_id)
        tiny_unet_path = _find_tiny_sd15_unet(local)
        if not tiny_unet_path:
            log.error("[sd] tiny-sd15 UNet missing; fallback unlikely to work")

    # TEXT MODE
    if mode == "text" and entry["pipe_text"] is None:
        local = _download_model(model_id)
        model_index = _load_model_index(local)

        # 1) try architecture-driven loader
        arch_pipe = _bootstrap_from_model_index(local, device)
        if arch_pipe:
            entry["pipe_text"] = arch_pipe

        # 2) tiny pipeline
        elif is_tiny and tiny_unet_path:
            entry["pipe_text"] = _bootstrap_tiny_sd15_pipeline(
                StableDiffusionPipeline, tiny_unet_path, device
            )

        # 3) direct-from-hub
        elif use_direct:
            entry["pipe_text"] = _init_pipeline(
                StableDiffusionPipeline, model_id, device, local=False
            )

        # 4) standard SD layout
        else:
            if _has_model_index(local):
                entry["pipe_text"] = _init_pipeline(
                    StableDiffusionPipeline, local, device, local=True
                )
            else:
                entry["pipe_text"] = _init_pipeline(
                    StableDiffusionPipeline, model_id, device, local=False
                )

    # IMG2IMG MODE
    if mode == "img2img" and entry["pipe_img2img"] is None:
        local = _download_model(model_id)
        model_index = _load_model_index(local)

        # If architecture is TextToVideo, reuse same pipeline
        arch_pipe = _bootstrap_from_model_index(local, device)
        if arch_pipe:
            entry["pipe_img2img"] = archive_pipe = arch_pipe
        elif is_tiny and tiny_unet_path:
            entry["pipe_img2img"] = _bootstrap_tiny_sd15_pipeline(
                StableDiffusionImg2ImgPipeline, tiny_unet_path, device
            )
        elif use_direct:
            entry["pipe_img2img"] = _init_pipeline(
                StableDiffusionImg2ImgPipeline, model_id, device, local=False
            )
        else:
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
