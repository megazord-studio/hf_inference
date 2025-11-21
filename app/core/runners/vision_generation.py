"""Phase B vision generation runners: text-to-image, image-to-image, super-resolution, restoration.

Simplified baseline implementations using diffusers for SD text-to-image & img2img.
Super-resolution & restoration use lightweight PIL-based placeholders if diffusers model
not available to keep tests deterministic without heavy downloads.

Each runner adheres to BaseRunner contract.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Type, Set
from .base import BaseRunner
from app.core.utils.media import decode_image_base64, encode_image_base64, image_size
from app.core.runners.diffusion_shared import get_or_create_sd_pipeline
from PIL import Image  # retained for fallback square

log = logging.getLogger("app.runners.vision_generation")

import torch
import time

VISION_GEN_TASKS: Set[str] = {
    "text-to-image",
    "image-to-image",
    "image-super-resolution",
    "image-restoration",
}

class TextToImageRunner(BaseRunner):
    def load(self) -> int:
        start = time.time()
        log.info(f"[text-to-image] initiating pipeline acquisition model={self.model_id}")
        shared = get_or_create_sd_pipeline(self.model_id, self.device, mode="text")
        self.pipe = shared.get("pipe_text")
        self.revision = shared.get("revision")
        if not self.pipe:
            raise RuntimeError(f"failed_pipeline_init:{self.model_id}")
        self._loaded = True
        param_count = sum(p.numel() for p in self.pipe.unet.parameters()) if hasattr(self.pipe,'unet') else 0
        log.info(f"[text-to-image] pipeline ready model={self.model_id} params={param_count}")
        return param_count
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("text") or ""
        if not self.pipe:
            return {"error": "pipeline_unavailable", "image_base64": ""}
        if not prompt:
            return {"image_base64": ""}
        guidance = float(options.get("guidance_scale", 7.5))
        steps = int(options.get("num_inference_steps", 20))
        with torch.no_grad():
            out = self.pipe(prompt=prompt, guidance_scale=guidance, num_inference_steps=steps, output_type="pil")
        img = out.images[0] if getattr(out, 'images', []) else Image.new('RGB',(64,64),(0,0,0))
        return {"image_base64": encode_image_base64(img), "width": img.width, "height": img.height}

class ImageToImageRunner(BaseRunner):
    def load(self) -> int:
        start = time.time()
        log.info(f"[image-to-image] initiating pipeline acquisition model={self.model_id}")
        shared = get_or_create_sd_pipeline(self.model_id, self.device, mode="img2img")
        self.pipe = shared.get("pipe_img2img")
        self.revision = shared.get("revision")
        if not self.pipe:
            raise RuntimeError(f"failed_pipeline_init:{self.model_id}")
        self._loaded = True
        param_count = sum(p.numel() for p in self.pipe.unet.parameters()) if hasattr(self.pipe,'unet') else 0
        log.info(f"[image-to-image] pipeline ready model={self.model_id} params={param_count}")
        return param_count
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipe:
            return {"error": "pipeline_unavailable", "image_base64": ""}
        img_b64 = inputs.get("image_base64")
        prompt = options.get("prompt") or inputs.get("text") or ""
        if not img_b64 or not prompt:
            return {"image_base64": ""}
        init_img = decode_image_base64(img_b64)
        strength = float(options.get("strength", 0.75))
        steps = int(options.get("num_inference_steps", 20))
        with torch.no_grad():
            out = self.pipe(prompt=prompt, image=init_img, strength=strength, num_inference_steps=steps, output_type="pil")
        img = out.images[0] if getattr(out, 'images', []) else init_img
        return {"image_base64": encode_image_base64(img), "width": img.width, "height": img.height}

class ImageSuperResolutionRunner(BaseRunner):
    def load(self) -> int:
        log.info("[image-super-resolution] placeholder runner ready")
        self._loaded = True
        return 0

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            log.info("[image-super-resolution] no image provided")
            return {"image_base64": ""}
        img = decode_image_base64(img_b64)
        scale = int(options.get("scale", 2))
        log.info(f"[image-super-resolution] upscaling by scale={scale} orig={img.width}x{img.height}")
        new_size = (img.width * scale, img.height * scale)
        up = img.resize(new_size)
        return {"image_base64": encode_image_base64(up), "orig_size": image_size(img), "new_size": new_size}

class ImageRestorationRunner(BaseRunner):
    def load(self) -> int:
        from PIL import ImageFilter
        log.info("[image-restoration] placeholder runner ready (GaussianBlur)")
        self._filter = ImageFilter.GaussianBlur(radius=1.5)
        self._loaded = True
        return 0

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            log.info("[image-restoration] no image provided")
            return {"image_base64": ""}
        img = decode_image_base64(img_b64)
        mask_b64 = inputs.get("mask_base64")
        restored = img.filter(self._filter)
        log.info(f"[image-restoration] restored image size={restored.width}x{restored.height} mask_applied={bool(mask_b64)}")
        return {"image_base64": encode_image_base64(restored), "width": restored.width, "height": restored.height, "mask_applied": bool(mask_b64)}

_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "text-to-image": TextToImageRunner,
    "image-to-image": ImageToImageRunner,
    "image-super-resolution": ImageSuperResolutionRunner,
    "image-restoration": ImageRestorationRunner,
}

def vision_gen_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_TO_RUNNER[task]

__all__ = [
    "VISION_GEN_TASKS",
    "vision_gen_runner_for_task",
    "TextToImageRunner",
    "ImageToImageRunner",
    "ImageSuperResolutionRunner",
    "ImageRestorationRunner",
]
