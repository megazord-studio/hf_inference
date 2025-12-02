from __future__ import annotations

import logging
import time
from typing import Any
from typing import Dict
from typing import Set
from typing import Type

import torch

from app.core.runners.diffusion_shared import get_or_create_sd_pipeline
from app.core.utils.media import decode_image_base64
from app.core.utils.media import encode_image_base64
from app.core.utils.media import image_size

from .base import BaseRunner

log = logging.getLogger("app.runners.vision_generation")

VISION_GEN_TASKS: Set[str] = {
    "text-to-image",
    "image-to-image",
    "image-super-resolution",
    "image-restoration",
}


class TextToImageRunner(BaseRunner):
    def load(self) -> int:
        log.info("vision_generation: loading model_id=%s", self.model_id)
        shared = get_or_create_sd_pipeline(
            self.model_id, self.device, mode="text"
        )
        self.pipe = shared.get("pipe_text")
        if not self.pipe:
            raise RuntimeError(f"text_to_image_init_failed:{self.model_id}")
        self._loaded = True
        return (
            sum(p.numel() for p in self.pipe.unet.parameters())
            if hasattr(self.pipe, "unet")
            else 0
        )

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.pipe:
            raise RuntimeError("text_to_image_pipeline_unavailable")
        # Accept both 'prompt' and legacy 'text'
        prompt = inputs.get("prompt") or inputs.get("text") or ""
        if not prompt:
            raise RuntimeError("text_to_image_missing_prompt")
        guidance = float(options.get("guidance_scale", 7.5))
        steps = int(options.get("num_inference_steps", 20))
        stream = bool(options.get("_stream", False))
        start = time.time()
        with torch.no_grad():
            out = self.pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                output_type="pil",
            )
        runtime_ms = int((time.time() - start) * 1000)
        if not getattr(out, "images", None):
            raise RuntimeError("text_to_image_no_images")
        img = out.images[0]
        payload: Dict[str, Any] = {"image_base64": encode_image_base64(img)}
        if stream:
            payload["runtime_ms"] = runtime_ms
            payload["num_inference_steps"] = steps
        return payload


class ImageToImageRunner(BaseRunner):
    def load(self) -> int:
        log.info("vision_generation: loading model_id=%s", self.model_id)
        shared = get_or_create_sd_pipeline(
            self.model_id, self.device, mode="img2img"
        )
        self.pipe = shared.get("pipe_img2img")
        if not self.pipe:
            raise RuntimeError(f"image_to_image_init_failed:{self.model_id}")
        self._loaded = True
        return (
            sum(p.numel() for p in self.pipe.unet.parameters())
            if hasattr(self.pipe, "unet")
            else 0
        )

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.pipe:
            raise RuntimeError("image_to_image_pipeline_unavailable")
        img_b64 = inputs.get("image_base64")
        prompt = options.get("prompt") or inputs.get("text") or ""
        if not img_b64 or not prompt:
            raise RuntimeError("image_to_image_missing_inputs")
        init_img = decode_image_base64(img_b64)
        strength = float(options.get("strength", 0.75))
        steps = int(options.get("num_inference_steps", 20))
        stream = bool(options.get("_stream", False))
        start = time.time()
        with torch.no_grad():
            out = self.pipe(
                prompt=prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                output_type="pil",
            )
        runtime_ms = int((time.time() - start) * 1000)
        if not getattr(out, "images", None):
            raise RuntimeError("image_to_image_no_images")
        img = out.images[0]
        payload: Dict[str, Any] = {"image_base64": encode_image_base64(img)}
        if stream:
            payload["runtime_ms"] = runtime_ms
            payload["num_inference_steps"] = steps
        return payload


class ImageSuperResolutionRunner(BaseRunner):
    def load(self) -> int:
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            return {"image_base64": ""}
        img = decode_image_base64(img_b64)
        scale = int(options.get("scale", 2))
        new_size = (img.width * scale, img.height * scale)
        up = img.resize(new_size)
        return {
            "image_base64": encode_image_base64(up),
            "orig_size": image_size(img),
            "new_size": new_size,
        }


class ImageRestorationRunner(BaseRunner):
    def load(self) -> int:
        from PIL import ImageFilter

        self._filter = ImageFilter.GaussianBlur(radius=1.5)
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            return {"image_base64": ""}
        img = decode_image_base64(img_b64)
        mask_b64 = inputs.get("mask_base64")
        restored = img.filter(self._filter)
        return {
            "image_base64": encode_image_base64(restored),
            "width": restored.width,
            "height": restored.height,
            "mask_applied": bool(mask_b64),
        }


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
