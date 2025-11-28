"""Gemma-specific input building and pipeline logic."""
from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Optional

from transformers import AutoProcessor
from transformers import pipeline

from .tokenizer import ensure_image_tokens
from .tokenizer import extract_text
from .tokenizer import get_tokenizer
from .utils import cap_max_new_tokens
from .utils import is_cuda
from .utils import move_to_device
from .utils import resolve_max_new_tokens
from .utils import safe_call

log = logging.getLogger("app.runners.multimodal")


def build_gemma_chat_messages(
    image: Any,
    question: str,
    num_images: int = 1,
) -> list:
    """Build Gemma chat-style messages with image placeholders."""
    token_count = max(1, int(num_images) if num_images else 1)
    prompt = (question or "").strip() or "Describe the image."
    content = []
    for _ in range(token_count):
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def try_gemma_pipeline(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model_id: str,
    processor: Any,
    device: Any,
) -> Optional[str]:
    """Try to run Gemma through image-text-to-text pipeline."""
    if "gemma" not in model_id.lower():
        return None

    dev = 0 if is_cuda(device) else None
    try:
        log.info("gemma: building image-text-to-text pipeline (device=%s)", dev)
        pl = pipeline(
            "image-text-to-text",
            model=model_id,
            trust_remote_code=True,
            device=dev,
        )
    except Exception as e:
        log.info("gemma pipeline build failed: %s", e)
        return None

    resolved, user_override = resolve_max_new_tokens(options, device, default=16)
    gen_max = resolved if user_override else cap_max_new_tokens(resolved, device)
    prompt = format_gemma_chat_prompt(question, 1, processor, model_id)
    if not prompt:
        tokenizer = get_tokenizer(processor=processor)
        prompt = ensure_image_tokens(question or "", 1, tokenizer)

    try:
        log.info("gemma: calling pipeline with max_new_tokens=%d (user_override=%s)", gen_max, user_override)
        result_any = pl(
            images=[image],
            text=prompt,
            generate_kwargs={
                "max_new_tokens": gen_max,
                "do_sample": False,
                "num_beams": 1,
            },
        )
        text = extract_text(result_any)
        log.info(
            "gemma pipeline returned: %s",
            (text[:80] + "...") if isinstance(text, str) and len(text) > 80 else text,
        )
        if text:
            return text
    except Exception as e:
        log.info("gemma pipeline call failed: %s", e)

    return None


def build_gemma_generation_inputs(
    image: Any,
    question: str,
    processor: Any,
    model_id: str,
    device: Any,
) -> Optional[Dict[str, Any]]:
    """Build generation inputs for Gemma models."""
    if processor is None:
        log.info("gemma: processor unavailable, skipping generation inputs")
        return None

    prompt = format_gemma_chat_prompt(question, 1, processor, model_id)
    if not prompt:
        tokenizer = get_tokenizer(processor=processor)
        prompt = ensure_image_tokens(question or "", 1, tokenizer)

    try:
        enc = processor(images=[image], text=prompt, return_tensors="pt")
        return move_to_device(enc, device)
    except Exception as e:
        log.info("gemma: processor encoding failed: %s", e)
        return None


def format_gemma_chat_prompt(
    question: str,
    num_images: int,
    processor: Any,
    model_id: str,
) -> Optional[str]:
    """Format question using Gemma chat template."""
    if "gemma" not in model_id.lower():
        return None
    if processor is None:
        return None

    messages = build_gemma_chat_messages(None, question, num_images=num_images)
    apply_template = getattr(processor, "apply_chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    template_fn = apply_template or getattr(tokenizer, "apply_chat_template", None)

    if callable(template_fn):
        try:
            return template_fn(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            log.info("gemma chat template failed: %s", e)

    return None


def ensure_processor_loaded(model_id: str, current_processor: Any) -> Any:
    """Ensure processor is loaded, loading if necessary."""
    if current_processor is not None:
        return current_processor
    return safe_call(
        lambda: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    )


def build_vlm_inputs(
    image: Any,
    question: str,
    options: Dict[str, Any],
    processor: Any,
    model: Any,
    device: Any,
    model_id: str,
) -> Dict[str, Any]:
    """Build inputs for VLM prediction."""
    # Try Gemma pipeline first
    txt = try_gemma_pipeline(image, question, options, model_id, processor, device)
    if txt is not None:
        log.info("gemma pipeline produced text; short-circuit")
        return {"_pipeline_text": txt}

    # Build Gemma-specific inputs
    if "gemma" in model_id.lower():
        processor = ensure_processor_loaded(model_id, processor)
        gemma_enc = build_gemma_generation_inputs(
            image, question, processor, model_id, device
        )
        if gemma_enc is not None:
            return gemma_enc

    # Try encoding with processor
    enc = _encode_with_processor(image, question, processor, device)
    if enc is not None:
        return enc

    # Fallback to tokenizer
    return _encode_with_tokenizer(question, processor, model, device)


def _encode_with_processor(
    image: Any,
    question: str,
    processor: Any,
    device: Any,
) -> Optional[Dict[str, Any]]:
    """Try to encode with processor."""
    if processor is None:
        return None
    try:
        enc = processor(images=[image], text=question, return_tensors="pt")
        return move_to_device(enc, device)
    except Exception as e:
        log.debug("processor encoding failed: %s", e)
        return None


def _encode_with_tokenizer(
    question: str,
    processor: Any,
    model: Any,
    device: Any,
) -> Dict[str, Any]:
    """Encode with tokenizer as fallback."""
    try:
        tokenizer = get_tokenizer(processor=processor, model=model)
        enc = tokenizer(question, return_tensors="pt")
        return move_to_device(enc, device)
    except Exception as e:
        log.debug("tokenizer encoding failed: %s", e)
        return {"_skip_generation": True}
