from __future__ import annotations

import logging
from typing import Any
from typing import Dict

import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

from app.core.runners.base import BaseRunner

from .utils import decode_base64_image

log = logging.getLogger("app.runners.vision")


class ImageClassificationRunner(BaseRunner):
    """Image classification runner using transformers."""

    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")

        try:
            log.info(
                "vision: loading AutoImageProcessor for %s", self.model_id
            )
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        except Exception:
            from transformers import AutoFeatureExtractor

            log.info(
                "vision: loading AutoFeatureExtractor for %s", self.model_id
            )
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_id
            )

        log.info(
            "vision: loading AutoModelForImageClassification for %s",
            self.model_id,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_id
        )

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
            return {"predictions": []}

        try:
            image = decode_base64_image(img_b64)
            encoded = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = self.model(
                    **{k: v.to(self.model.device) for k, v in encoded.items()}
                )
                probs = out.logits.softmax(-1)[0]

            requested_k = int(options.get("top_k", 3))
            top_k = max(1, min(requested_k, probs.shape[-1]))
            values, indices = probs.topk(top_k)

            labels = [self.model.config.id2label[i.item()] for i in indices]
            return {
                "predictions": [
                    {"label": lbl, "score": float(v.item())}
                    for lbl, v in zip(labels, values)
                ]
            }
        except Exception as e:
            log.warning("image-classification predict error: %s", e)
            return {"predictions": []}
