"""Stable Diffusion shared loader (full pipeline, no stubbing).
DRY/KISS: always download required snapshot (leverages HF cache), then init pipeline.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import os
import time
import json

import torch
from huggingface_hub import snapshot_download

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

log = logging.getLogger("app.runners.diffusion_shared")

_SD_CACHE: Dict[str, Dict[str, Any]] = {}

ALLOWED_CLIP_KEYS = {
    'architectures','model_type','text_config','vision_config','projection_dim','initializer_range','layer_norm_eps',
    'hidden_act','hidden_size','intermediate_size','max_position_embeddings','num_attention_heads','num_hidden_layers',
    'pad_token_id','bos_token_id','eos_token_id','vocab_size'
}

def _parse_model_id(model_id: str) -> tuple[str, Optional[str]]:
    if '@' in model_id:
        base, rev = model_id.split('@', 1)
        return base, rev or None
    return model_id, None


def _download_model(model_id: str) -> tuple[str, Optional[str]]:
    base_id, revision = _parse_model_id(model_id)
    start = time.time()
    log.info(f"[sd] download start model={base_id} revision={revision or 'latest'}")
    kwargs = {
        'repo_id': base_id,
        'max_workers': 16,
        'local_dir': None,
        'local_dir_use_symlinks': True,
    }
    if revision:
        kwargs['revision'] = revision
    local_dir = snapshot_download(**kwargs)
    # metrics
    total_bytes = 0; file_count = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            fp = os.path.join(root, f)
            try: total_bytes += os.path.getsize(fp)
            except Exception: pass
            file_count += 1
    dur = max(time.time() - start, 1e-6)
    mb = total_bytes / (1024**2)
    log.info(f"[sd] download done model={base_id} revision={revision or 'latest'} files={file_count} size_mb={mb:.2f} time_s={dur:.2f}")
    return local_dir, revision


def _init_pipeline(pipe_cls, local_dir: str, device: Optional[torch.device]):
    start = time.time()
    dtype = torch.float16 if (device and device.type in ("cuda", "mps")) else None
    log.info(f"[sd] init start model_dir={os.path.basename(local_dir)} dtype={'fp16' if dtype==torch.float16 else 'fp32'}")
    pipe = pipe_cls.from_pretrained(local_dir, local_files_only=True, torch_dtype=dtype)
    if device:
        try:
            pipe.to(device)
        except Exception:
            pass
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
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


def _sanitize_clip_config(local_dir: str) -> None:
    cfg_path = os.path.join(local_dir, 'text_encoder', 'config.json')
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        removed_any = False
        # Whitelist top-level keys
        filtered = {}
        for k, v in data.items():
            if k in ALLOWED_CLIP_KEYS or k.endswith('_token_id') or k.endswith('_size'):
                filtered[k] = v
            else:
                removed_any = True
        # Also sanitize nested 'text_config' if present
        if 'text_config' in filtered and isinstance(filtered['text_config'], dict):
            txt_cfg = filtered['text_config']
            for bad in list(txt_cfg.keys()):
                if bad not in ALLOWED_CLIP_KEYS and not bad.endswith('_token_id') and not bad.endswith('_size'):
                    txt_cfg.pop(bad, None); removed_any = True
        if removed_any:
            with open(cfg_path, 'w') as f:
                json.dump(filtered, f)
            log.info(f"[sd] sanitized text_encoder config (removed unsupported keys) in {cfg_path}")
    except Exception as e:
        log.info(f"[sd] sanitize skipped ({e})")


def get_or_create_sd_pipeline(model_id: str, device: Optional[torch.device], mode: str) -> Dict[str, Any]:
    base_id, revision = _parse_model_id(model_id)
    cache_key = f"{base_id}@{revision or 'latest'}"
    entry = _SD_CACHE.get(cache_key)
    if entry is None:
        entry = {"pipe_text": None, "pipe_img2img": None, "revision": revision}
        _SD_CACHE[cache_key] = entry
    if mode not in {"text", "img2img"}:
        raise ValueError("mode must be 'text' or 'img2img'")
    if mode == "text" and entry["pipe_text"] is None:
        local, _ = _download_model(model_id)
        _sanitize_clip_config(local)
        entry["pipe_text"] = _init_pipeline(StableDiffusionPipeline, local, device)
    if mode == "img2img" and entry["pipe_img2img"] is None:
        local, _ = _download_model(model_id)
        _sanitize_clip_config(local)
        entry["pipe_img2img"] = _init_pipeline(StableDiffusionImg2ImgPipeline, local, device)
    return entry

__all__ = ["get_or_create_sd_pipeline"]
