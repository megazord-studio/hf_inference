"""Model loaders for multimodal architectures.

Each loader follows a consistent pattern:
- Takes model_id, device, and necessary utilities
- Returns (model, processor, tokenizer, param_count)
- Uses safe_call for error handling
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Optional
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import BlipForQuestionAnswering
from transformers import BlipProcessor
from transformers import LlavaForConditionalGeneration
from transformers import LlavaProcessor
from transformers import pipeline

from .utils import count_params
from .utils import safe_call
from .utils import select_dtype
from .utils import to_device
from .utils import unify_model_dtype

log = logging.getLogger("app.runners.multimodal")


LoadResult = Tuple[Any, Any, Any, int]  # model, processor, tokenizer, param_count


def load_blip(model_id: str, device: Any) -> LoadResult:
    """Load BLIP VQA model."""
    log.info("multimodal: loading BLIP model_id=%s", model_id)
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor, None, count_params(model)


def load_llava(model_id: str, device: Any) -> LoadResult:
    """Load LLaVA model."""
    try:
        log.info("multimodal: loading LLaVA model_id=%s", model_id)
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model, processor, None, count_params(model)
    except Exception as e:
        log.error("llava load failed: %s", e)
        return None, None, None, 0


def load_qwen_vl(model_id: str, device: Any) -> LoadResult:
    """Load Qwen-VL model."""
    log.info("multimodal: loading Qwen-VL model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    )
    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, None, count_params(model)


def load_minicpm(model_id: str, device: Any) -> LoadResult:
    """Load MiniCPM-V model."""
    from transformers import AutoModel
    from transformers import AutoTokenizer

    load_dtype = select_dtype(device)
    log.info("multimodal: loading MiniCPM-V model_id=%s", model_id)

    tokenizer = safe_call(
        lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=load_dtype
        )
    )
    model = to_device(model, device)
    unify_model_dtype(model, load_dtype)

    if model is not None:
        _patch_minicpm_generate(model, tokenizer)
    if hasattr(model, "eval"):
        model.eval()

    return model, None, tokenizer, count_params(model)


def load_vlm(model_id: str, device: Any) -> Tuple[Any, Any, Any, int, str]:
    """Load generic VLM model. Returns (model, processor, tokenizer, params, arch)."""
    log.info("multimodal: loading VLM (auto) model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )

    # Try different model classes in order
    model = safe_call(
        lambda: AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True
        )
    )
    arch = "vlm"

    if model is None:
        model = safe_call(
            lambda: AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        )

    if model is None:
        model = safe_call(
            lambda: AutoModelForVision2Seq.from_pretrained(
                model_id, trust_remote_code=True
            )
        )
        if model is None:
            arch = "vlm_unsupported"

    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()

    return model, processor, None, count_params(model), arch


def load_generic_vqa(model_id: str, device: Any) -> LoadResult:
    """Load model using VQA pipeline."""
    log.info("multimodal: initializing VQA pipeline model_id=%s", model_id)
    dev = 0 if device and getattr(device, "type", None) == "cuda" else None
    pipe = pipeline(
        task="visual-question-answering",
        model=model_id,
        device=dev,
        trust_remote_code=True,
    )
    m = getattr(pipe, "model", None)
    return pipe, None, None, count_params(m)


def load_yi_vl(model_id: str, device: Any) -> Tuple[Any, Any, Any, int, str]:
    """Load Yi-VL model. Returns (model, processor, tokenizer, params, arch)."""
    log.info("multimodal: loading Yi-VL model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    )
    arch = "yi_vl"
    if model is None:
        log.warning("yi_vl: failed to load model, may be state_dict mismatch")
        arch = "yi_vl_unsupported"
        return None, processor, None, 0, arch

    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, None, count_params(model), arch


def load_internvl(model_id: str, device: Any) -> Tuple[Any, Any, Any, int, str]:
    """Load InternVL2 model."""
    log.info("multimodal: loading InternVL model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    )
    arch = "internvl"

    if model is None:
        model = safe_call(
            lambda: AutoModelForVision2Seq.from_pretrained(
                model_id, trust_remote_code=True
            )
        )
    if model is None:
        log.warning("internvl: failed to load model")
        return None, processor, None, 0, "internvl_unsupported"

    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, None, count_params(model), arch


def load_kosmos2(model_id: str, device: Any) -> Tuple[Any, Any, Any, int, str]:
    """Load Kosmos-2 model."""
    log.info("multimodal: loading Kosmos-2 model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
    )
    arch = "kosmos2"

    if model is None:
        model = safe_call(
            lambda: AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        )
    if model is None:
        log.warning("kosmos2: failed to load model")
        return None, processor, None, 0, "kosmos2_unsupported"

    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, None, count_params(model), arch


def load_florence2(model_id: str, device: Any) -> Tuple[Any, Any, Any, int, str]:
    """Load Florence-2 model."""
    log.info("multimodal: loading Florence-2 model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    )
    arch = "florence2"

    if model is None:
        model = safe_call(
            lambda: AutoModelForVision2Seq.from_pretrained(
                model_id, trust_remote_code=True
            )
        )
    if model is None:
        log.warning("florence2: failed to load model")
        return None, processor, None, 0, "florence2_unsupported"

    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, None, count_params(model), arch


def load_cogvlm(model_id: str, device: Any) -> Tuple[Any, Any, Any, int, str]:
    """Load CogVLM2 model."""
    log.info("multimodal: loading CogVLM model_id=%s", model_id)
    processor = safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )
    model = safe_call(
        lambda: AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    )
    arch = "cogvlm"

    if model is None:
        log.warning("cogvlm: failed to load model")
        return None, processor, None, 0, "cogvlm_unsupported"

    model = to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, None, count_params(model), arch


def _patch_minicpm_generate(model: Any, tokenizer: Optional[Any]) -> None:
    """Patch MiniCPM model with GenerationMixin for generate() support."""
    if model is None:
        return
    llm = getattr(model, "llm", None)
    if llm is None:
        return

    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:
        from transformers.generation import GenerationMixin

    if GenerationMixin not in llm.__class__.__mro__:
        Patched = type(
            llm.__class__.__name__ + "Patched",
            (GenerationMixin, llm.__class__),
            {},
        )
        llm.__class__ = Patched
        log.info("patched %s with GenerationMixin", llm.__class__.__name__)

    if not hasattr(llm, "generation_config") or llm.generation_config is None:
        try:
            from transformers import GenerationConfig
            from transformers import PretrainedConfig

            base = getattr(llm, "config", None)
            if isinstance(base, PretrainedConfig):
                llm.generation_config = GenerationConfig.from_model_config(base)
            else:
                llm.generation_config = GenerationConfig()
        except Exception:
            from transformers.generation import GenerationConfig

            llm.generation_config = GenerationConfig()

    if tokenizer is not None:
        for kid in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            val = getattr(tokenizer, kid, None)
            if val is None:
                continue
            if getattr(llm.generation_config, kid, None) is None:
                setattr(llm.generation_config, kid, val)
            cfg = getattr(llm, "config", None)
            if cfg is not None and getattr(cfg, kid, None) is None:
                try:
                    setattr(cfg, kid, val)
                except Exception:
                    pass
