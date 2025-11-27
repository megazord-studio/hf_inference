"""Tokenizer and text processing utilities for multimodal runners."""
from __future__ import annotations

import logging
from typing import Any
from typing import Optional

log = logging.getLogger("app.runners.multimodal")


def get_tokenizer(
    tokenizer: Any = None,
    processor: Any = None,
    model: Any = None,
) -> Any:
    """Get tokenizer from available sources."""
    if tokenizer is not None:
        return tokenizer
    if processor is not None and hasattr(processor, "tokenizer"):
        return processor.tokenizer
    if model is not None and hasattr(model, "tokenizer"):
        return model.tokenizer
    raise RuntimeError("no tokenizer/processor available")


def get_image_token(tokenizer: Any) -> str:
    """Get the image token from tokenizer."""
    im = getattr(tokenizer, "image_token", None)
    if isinstance(im, str) and im:
        return im
    stm = getattr(tokenizer, "special_tokens_map", {}) or {}
    for v in stm.values():
        if isinstance(v, str) and "image" in v.lower():
            return v
    addl = getattr(tokenizer, "additional_special_tokens", None)
    if isinstance(addl, (list, tuple)):
        for a in addl:
            if isinstance(a, str) and "image" in a.lower():
                return a
    return "<image>"


def ensure_image_tokens(question: str, num_images: int, tokenizer: Any) -> str:
    """Ensure the question has the correct number of image tokens."""
    token = get_image_token(tokenizer)
    if question.count(token) == num_images:
        return question
    q = question.replace(token, "").strip()
    return f"{' '.join([token] * num_images)} {q}".strip()


def decode_output(out: Any, processor: Any, tokenizer: Any) -> str:
    """Decode model output to text."""
    if out is None or (hasattr(out, "__len__") and len(out) == 0):
        return ""
    if processor is not None:
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(out, skip_special_tokens=True)
            return decoded[0] if decoded else ""
        if hasattr(processor, "decode"):
            return processor.decode(out[0], skip_special_tokens=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_text(out: Any) -> Optional[str]:
    """Extract text from various output formats."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return out.get("text") or out.get("generated_text") or out.get("answer")
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            return (
                first.get("generated_text")
                or first.get("text")
                or first.get("answer")
            )
        if isinstance(first, list) and first and isinstance(first[0], dict):
            return first[0].get("generated_text") or first[0].get("text")
    return None


def strip_processor_only_kwargs(enc: dict, model_id: str) -> None:
    """Remove processor-only kwargs that can't be passed to generate()."""
    if "gemma" in model_id.lower():
        enc.pop("num_crops", None)
