from __future__ import annotations

import gc
import logging
from typing import Any
from typing import Dict
from typing import Generator
from typing import Optional

import torch

from app.core.runners.base import BaseRunner
from app.core.utils.media import decode_image_base64

from .arch import detect_arch
from .loaders import load_blip
from .loaders import load_cogvlm
from .loaders import load_florence2
from .loaders import load_generic_vqa
from .loaders import load_internvl
from .loaders import load_kosmos2
from .loaders import load_llava
from .loaders import load_minicpm
from .loaders import load_qwen_vl
from .loaders import load_vlm
from .loaders import load_yi_vl
from .predictors import predict_blip
from .predictors import predict_cogvlm
from .predictors import predict_florence2
from .predictors import predict_generic_vqa
from .predictors import predict_internvl
from .predictors import predict_kosmos2
from .predictors import predict_llava
from .predictors import predict_minicpm
from .predictors import predict_qwen_vl
from .predictors import predict_vlm
from .predictors import predict_yi_vl
from .utils import cap_max_new_tokens
from .utils import require
from .utils import resolve_max_new_tokens

log = logging.getLogger("app.runners.multimodal")


class ImageTextToTextRunner(BaseRunner):
    """Image+Text -> Text multimodal runner with architecture-specific flows."""

    def __init__(
        self, model_id: str, device: Any, dtype: Optional[str] = None
    ) -> None:
        super().__init__(model_id, device)
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.pipe: Any = None
        self._arch: Optional[str] = None

    def unload(self) -> None:
        """Release model, processor, tokenizer, pipeline to free memory."""
        for attr in ("model", "processor", "tokenizer", "pipe"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> int:
        """Load model based on detected architecture."""
        self._arch = detect_arch(self.model_id)
        return self._load_by_arch()

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run prediction using architecture-specific predictor."""
        image = decode_image_base64(
            require(inputs.get("image_base64"), "missing_image")
        )
        question = (
            inputs.get("text") or options.get("question") or "What is shown?"
        )
        log.info(
            "multimodal.predict start model_id=%s arch=%s",
            self.model_id,
            self._arch,
        )

        return self._predict_by_arch(image, question, options)

    def predict_stream(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream prediction (only supported for BLIP)."""
        if self._arch != "blip":
            yield {"event": "error", "data": "streaming_not_supported"}
            return

        image = decode_image_base64(
            require(inputs.get("image_base64"), "missing_image")
        )
        question = (
            inputs.get("text") or options.get("question") or "What is shown?"
        )
        if self.processor is None:
            yield {"event": "error", "data": "processor_not_loaded"}
            return
        enc = self.processor(image, question, return_tensors="pt").to(
            self.device
        )
        requested, user_override = resolve_max_new_tokens(
            options, self.device, default=32
        )
        max_len = (
            requested
            if user_override
            else cap_max_new_tokens(requested, self.device)
        )

        if self.model is None:
            yield {"event": "error", "data": "model_not_loaded"}
            return
        with torch.no_grad():
            seq = self.model.generate(
                **enc, max_length=max_len, do_sample=False, num_beams=1
            )

        text = self.processor.decode(seq[0], skip_special_tokens=True)
        for i in range(0, len(text), 8):
            yield {"event": "token", "data": text[i : i + 8]}
        yield {"event": "done", "data": text}

    def _load_by_arch(self) -> int:
        """Route to appropriate loader based on architecture."""
        arch = self._arch or ""
        loaders: Dict[str, Any] = {
            "blip": self._load_blip,
            "llava": self._load_llava,
            "qwen_vl": self._load_qwen_vl,
            "minicpm_vlm": self._load_minicpm,
            "yi_vl": self._load_yi_vl,
            "internvl": self._load_internvl,
            "kosmos2": self._load_kosmos2,
            "florence2": self._load_florence2,
            "cogvlm": self._load_cogvlm,
            "vlm": self._load_vlm,
        }
        loader = loaders.get(arch, self._load_generic_vqa)
        return loader()

    def _predict_by_arch(
        self, image: Any, question: str, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route to appropriate predictor based on architecture."""
        arch = self._arch or ""
        predictors: Dict[str, Any] = {
            "blip": lambda: predict_blip(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
            ),
            "llava": lambda: predict_llava(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
            ),
            "qwen_vl": lambda: predict_qwen_vl(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                self.model_id,
            ),
            "minicpm_vlm": lambda: predict_minicpm(
                image,
                question,
                options,
                self.model,
                self.tokenizer,
                self.device,
            ),
            "yi_vl": lambda: predict_yi_vl(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                arch,
            ),
            "internvl": lambda: predict_internvl(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                arch,
            ),
            "kosmos2": lambda: predict_kosmos2(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                arch,
                self.model_id,
            ),
            "florence2": lambda: predict_florence2(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                arch,
                self.model_id,
            ),
            "cogvlm": lambda: predict_cogvlm(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                arch,
            ),
            "vlm": lambda: predict_vlm(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                self.model_id,
            ),
            "vlm_unsupported": lambda: predict_vlm(
                image,
                question,
                options,
                self.model,
                self.processor,
                self.device,
                self.model_id,
            ),
            "generic_vqa": lambda: predict_generic_vqa(
                image, question, options, self.pipe
            ),
        }
        predictor = predictors.get(arch, lambda: {})
        return predictor()

    # Loader wrappers that store results on self
    def _load_blip(self) -> int:
        self.model, self.processor, _, params = load_blip(
            self.model_id, self.device
        )
        return params

    def _load_llava(self) -> int:
        self.model, self.processor, _, params = load_llava(
            self.model_id, self.device
        )
        return params

    def _load_qwen_vl(self) -> int:
        self.model, self.processor, _, params = load_qwen_vl(
            self.model_id, self.device
        )
        return params

    def _load_minicpm(self) -> int:
        self.model, _, self.tokenizer, params = load_minicpm(
            self.model_id, self.device
        )
        return params

    def _load_yi_vl(self) -> int:
        self.model, self.processor, _, params, arch = load_yi_vl(
            self.model_id, self.device
        )
        self._arch = arch
        return params

    def _load_internvl(self) -> int:
        self.model, self.processor, _, params, arch = load_internvl(
            self.model_id, self.device
        )
        self._arch = arch
        return params

    def _load_kosmos2(self) -> int:
        self.model, self.processor, _, params, arch = load_kosmos2(
            self.model_id, self.device
        )
        self._arch = arch
        return params

    def _load_florence2(self) -> int:
        self.model, self.processor, _, params, arch = load_florence2(
            self.model_id, self.device
        )
        self._arch = arch
        return params

    def _load_cogvlm(self) -> int:
        self.model, self.processor, _, params, arch = load_cogvlm(
            self.model_id, self.device
        )
        self._arch = arch
        return params

    def _load_vlm(self) -> int:
        self.model, self.processor, _, params, arch = load_vlm(
            self.model_id, self.device
        )
        self._arch = arch
        return params

    def _load_generic_vqa(self) -> int:
        self.pipe, _, _, params = load_generic_vqa(self.model_id, self.device)
        return params
