"""Image Captioning runner."""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

from app.core.runners.base import BaseRunner

from .utils import decode_base64_image

log = logging.getLogger("app.runners.vision")


class ImageCaptioningRunner(BaseRunner):
    """Image captioning runner using VisionEncoderDecoder or CausalLM."""

    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")

        log.info("vision: loading AutoProcessor for %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Attempt VisionEncoderDecoderModel first (common for vit-gpt2)
        self._is_ved = False
        try:
            from transformers import VisionEncoderDecoderModel

            log.info(
                "vision: loading VisionEncoderDecoderModel for %s",
                self.model_id,
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id
            )
            self._is_ved = True
        except Exception:
            log.info(
                "vision: loading AutoModelForCausalLM for %s", self.model_id
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        if self.device:
            self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            return {"text": ""}

        try:
            image = decode_base64_image(img_b64)
            prompt = options.get("prompt", "")
            max_new = int(options.get("max_new_tokens", 30))

            if self._is_ved:
                enc = self.processor(images=image, return_tensors="pt")
                enc = {
                    k: (v.to(self.model.device) if hasattr(v, "to") else v)
                    for k, v in enc.items()
                }
                with torch.no_grad():
                    gen = self.model.generate(**enc, max_new_tokens=max_new)
                text = self.processor.batch_decode(
                    gen, skip_special_tokens=True
                )[0].strip()
                return {"text": text}

            # Fallback causal LM path
            encoded = self.processor(
                images=image, text=prompt, return_tensors="pt"
            )
            encoded = {
                k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in encoded.items()
            }

            with torch.no_grad():
                if "input_ids" in encoded:
                    gen = self.model.generate(
                        **encoded, max_new_tokens=max_new
                    )
                else:
                    gen = self.model.generate(
                        pixel_values=encoded.get("pixel_values"),
                        max_new_tokens=max_new,
                    )

            text = self.processor.batch_decode(gen, skip_special_tokens=True)[
                0
            ].strip()
            return {"text": text}
        except Exception as e:
            log.warning("image-captioning predict error: %s", e)
            return {"text": ""}
