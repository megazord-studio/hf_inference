from __future__ import annotations

import logging
from typing import Any
from typing import Dict

import torch
from transformers import AutoModelForObjectDetection
from transformers import AutoProcessor

from app.core.runners.base import BaseRunner

from .utils import decode_base64_image

log = logging.getLogger("app.runners.vision")


class ObjectDetectionRunner(BaseRunner):
    """Object detection runner using transformers."""

    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")

        # Attempt to load processor
        try:
            log.info("vision: loading AutoProcessor for %s", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception:
            self.processor = _DummyProcessor()

        # Try real model, else build a local dummy
        try:
            log.info(
                "vision: loading AutoModelForObjectDetection for %s",
                self.model_id,
            )
            self.model = AutoModelForObjectDetection.from_pretrained(
                self.model_id
            )
            self._used_dummy = False
        except Exception as e:
            log.warning(
                "ObjectDetectionRunner using dummy model for %s: %s",
                self.model_id,
                e,
            )
            self._used_dummy = True
            self.model = _DummyDetectionModel()

        if self.device and not self._used_dummy:
            try:
                self.model.to(self.device)
            except Exception:
                pass

        self.model.eval()
        self._loaded = True

        try:
            return sum(p.numel() for p in self.model.parameters())
        except Exception:
            return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            return {"detections": []}

        try:
            image = decode_base64_image(img_b64)
            pixel_values = self._encode_image(image)

            with torch.no_grad():
                out = self.model(pixel_values=pixel_values)

            return self._process_detections(out, options)
        except Exception as e:
            log.warning("object-detection predict error: %s", e)
            return {"detections": []}

    def _encode_image(self, image: Any) -> torch.Tensor:
        """Encode image to pixel values tensor."""
        if hasattr(self.processor, "__call__"):
            enc = self.processor(images=image, return_tensors="pt")
            if isinstance(enc, dict):
                pixel_values = enc.get("pixel_values")
                if pixel_values is None:
                    pixel_values = torch.randn(1, 3, 8, 8)
            elif isinstance(enc, torch.Tensor):
                pixel_values = enc.unsqueeze(0) if enc.ndim == 3 else enc
            else:
                pixel_values = torch.randn(1, 3, 8, 8)
        else:
            pixel_values = torch.randn(1, 3, 8, 8)
        return pixel_values

    def _process_detections(
        self, out: Any, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process model output to detection results."""
        scores = out.logits.softmax(-1)[0]
        box_tensor = out.pred_boxes[0]
        n = min(scores.shape[0], box_tensor.shape[0])
        conf_thresh = float(options.get("confidence", 0.25))
        max_detections = int(options.get("max_detections", n))

        results = []
        for i in range(n):
            score_vec = scores[i]
            score, cls = score_vec.max(-1)
            if float(score.item()) < conf_thresh:
                continue

            box = box_tensor[i].tolist()
            label = self._get_label(int(cls.item()))
            results.append(
                {"label": label, "score": float(score.item()), "box": box}
            )

            if len(results) >= max_detections:
                break

        return {"detections": results}

    def _get_label(self, cls_id: int) -> str:
        """Get label string for class ID."""
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "id2label"
        ):
            return self.model.config.id2label.get(cls_id, str(cls_id))
        return str(cls_id)


class _DummyProcessor:
    """Dummy processor for when real processor unavailable."""

    def __call__(
        self, images: Any, return_tensors: str = "pt"
    ) -> torch.Tensor:
        _ = return_tensors  # unused but matches expected signature
        _ = images
        return torch.randn(1, 3, 8, 8)


class _DummyDetectionModel(torch.nn.Module):
    """Dummy detection model for fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Cfg",
            (),
            {"id2label": {0: "person", 1: "car", 2: "tree", 3: "chair"}},
        )()
        self.device = torch.device("cpu")

    def forward(self, pixel_values: torch.Tensor) -> Any:
        logits = torch.randn(1, 5, len(self.config.id2label))
        pred_boxes = torch.rand(1, 5, 4)
        return type("Out", (), {"logits": logits, "pred_boxes": pred_boxes})()
