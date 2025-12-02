from __future__ import annotations

import inspect
import json
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation

from app.core.runners.base import BaseRunner

from .utils import decode_base64_image

segformer_cls: Optional[Type[Any]]
try:
    from transformers import (
        SegformerImageProcessor as _SegformerImageProcessor,
    )

    segformer_cls = _SegformerImageProcessor
except Exception:  # pragma: no cover
    segformer_cls = None

log = logging.getLogger("app.runners.vision")


class ImageSegmentationRunner(BaseRunner):
    """Semantic segmentation runner using transformers."""

    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        # Try standard transformers model + processor; on failure fallback to pipeline
        try:
            self._load_processor()
            log.info(
                "vision: loading AutoModelForSemanticSegmentation for %s",
                self.model_id,
            )
            try:
                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            except TypeError:
                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.model_id
                )
            if self.device:
                self.model.to(self.device)
            self.model.eval()
            self._backend = "transformers"
            self._loaded = True
            return sum(p.numel() for p in self.model.parameters())
        except Exception as e:
            log.info(
                "vision: falling back to pipeline(image-segmentation) for %s due to: %s",
                self.model_id,
                e,
            )
            from transformers import pipeline as hf_pipeline

            try:
                self.pipe = hf_pipeline(
                    task="image-segmentation",
                    model=self.model_id,
                    trust_remote_code=True,
                )
            except TypeError:
                self.pipe = hf_pipeline(
                    task="image-segmentation", model=self.model_id
                )
            self._backend = "pipeline"
            self._loaded = True
            return 0

    def _load_processor(self) -> None:
        """Load appropriate processor for segmentation."""
        processor_loaded = False
        # Special handling for Segformer models
        if segformer_cls is not None and "segformer" in self.model_id.lower():
            processor_loaded = self._try_load_segformer_processor(
                segformer_cls
            )
        if not processor_loaded:
            # Prefer AutoProcessor with remote code
            try:
                from transformers import AutoProcessor

                log.info(
                    "vision: loading AutoProcessor (trust_remote_code) for %s",
                    self.model_id,
                )
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
                except TypeError:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_id
                    )
                return
            except Exception:
                pass
            try:
                log.info(
                    "vision: loading AutoImageProcessor (trust_remote_code) for %s",
                    self.model_id,
                )
                try:
                    self.processor = AutoImageProcessor.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
                except TypeError:
                    self.processor = AutoImageProcessor.from_pretrained(
                        self.model_id
                    )
            except Exception:
                from transformers import AutoFeatureExtractor

                log.info(
                    "vision: loading AutoFeatureExtractor for %s",
                    self.model_id,
                )
                try:
                    self.processor = AutoFeatureExtractor.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
                except TypeError:
                    self.processor = AutoFeatureExtractor.from_pretrained(
                        self.model_id
                    )

    def _try_load_segformer_processor(self, segformer_cls: Any) -> bool:
        """Try to load SegformerImageProcessor with custom config handling."""
        if segformer_cls is None:  # guard import failure
            return False
        allowed_params = set(
            inspect.signature(segformer_cls.__init__).parameters.keys()
        ) - {"self"}
        strip_keys = {
            "reduce_labels",
            "feature_extractor_type",
            "image_processor_type",
        }
        config_files = [
            "preprocessor_config.json",
            "image_processor_config.json",
            "feature_extractor_config.json",
        ]
        for fname in config_files:
            try:
                log.info("vision: downloading %s for %s", fname, self.model_id)
                cfg_path = hf_hub_download(self.model_id, fname)
                with open(cfg_path, "r") as f:
                    raw_cfg = json.load(f)
                cfg = {
                    k: v
                    for k, v in raw_cfg.items()
                    if k in allowed_params and k not in strip_keys
                }
                self.processor = segformer_cls(**cfg)
                return True
            except Exception:
                continue
        return False

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            return {"labels": {}, "shape": []}
        try:
            image = decode_base64_image(img_b64)
            backend = getattr(self, "_backend", "transformers")
            if backend == "pipeline" and hasattr(self, "pipe"):
                res = self.pipe(image)
                shape = []
                try:
                    if isinstance(res, list) and res and isinstance(
                        res[0].get("mask"), (np.ndarray,)
                    ):
                        m = res[0]["mask"]
                        shape = list(getattr(m, "shape", []))
                    elif isinstance(res, list) and res and hasattr(
                        res[0].get("mask"), "size"
                    ):
                        m = res[0]["mask"]
                        w, h = m.size
                        shape = [h, w]
                except Exception:
                    shape = []
                return {"labels": {}, "shape": shape}
            # transformers backend
            enc = self.processor(images=image, return_tensors="pt").to(
                self.model.device
            )
            with torch.no_grad():
                out = self.model(**enc)
            logits = out.logits
            label_map = logits.argmax(1)[0].cpu().numpy()
            counts: Dict[str, int] = {}
            for lbl in np.unique(label_map):
                mask = label_map == lbl
                label_name = self.model.config.id2label.get(
                    int(lbl), str(int(lbl))
                )
                counts[label_name] = int(np.count_nonzero(mask))
            return {"labels": counts, "shape": list(label_map.shape)}
        except Exception as e:
            log.warning("image-segmentation predict error: %s", e)
            return {"labels": {}, "shape": []}
