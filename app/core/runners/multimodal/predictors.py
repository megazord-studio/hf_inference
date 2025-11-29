"""Prediction functions for multimodal architectures.

Each predictor follows a consistent pattern:
- Takes image, question, options, and model components
- Returns Dict[str, Any] with answer and arch fields
- Handles errors gracefully, returning {} on failure
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict

import torch

from .tokenizer import decode_output
from .tokenizer import ensure_image_tokens
from .tokenizer import get_tokenizer
from .tokenizer import strip_processor_only_kwargs
from .utils import cap_max_new_tokens
from .utils import move_to_device
from .utils import require
from .utils import resolve_max_new_tokens

log = logging.getLogger("app.runners.multimodal")


def predict_blip(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
) -> Dict[str, Any]:
    """Predict using BLIP model."""
    enc = processor(image, question, return_tensors="pt").to(device)
    resolved, user_override = resolve_max_new_tokens(
        options, device, default=32
    )
    max_len = (
        resolved if user_override else cap_max_new_tokens(resolved, device)
    )
    with torch.no_grad():
        out = model.generate(
            **enc, max_length=max_len, do_sample=False, num_beams=1
        )
    answer = processor.decode(out[0], skip_special_tokens=True)
    return {"answer": answer, "arch": "blip"}


def predict_llava(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
) -> Dict[str, Any]:
    """Predict using LLaVA model."""
    try:
        tokenizer = get_tokenizer(processor=processor, model=model)
        question = ensure_image_tokens(question, 1, tokenizer)
        enc = processor(images=image, text=question, return_tensors="pt").to(
            device
        )
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )

        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(out, skip_special_tokens=True)[0]
        elif hasattr(processor, "decode"):
            text = processor.decode(out[0], skip_special_tokens=True)
        else:
            text = tokenizer.decode(out[0], skip_special_tokens=True)

        return {"answer": text, "arch": "llava"}
    except Exception as e:
        log.error("llava generate failed: %s", e)
        return {}


def predict_qwen_vl(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    model_id: str,
) -> Dict[str, Any]:
    """Predict using Qwen-VL model."""
    import transformers as _tf

    from .utils import safe_call
    from .utils import to_device

    if model is None:
        model = safe_call(
            lambda: _tf.AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        )
        model = to_device(model, device)
        if hasattr(model, "eval"):
            model.eval()
        if model is None:
            return {}

    chat = getattr(model, "chat", None)
    if callable(chat):
        try:
            resp = (
                chat(processor, image, question)
                if processor is not None
                else chat(image, question)
            )
            return {"answer": str(resp), "arch": "qwen_vl"}
        except Exception as e:
            log.error("qwen_vl chat failed: %s", e)

    tok = (processor or model).tokenizer
    enc = {
        k: (v.to(device) if hasattr(v, "to") and device else v)
        for k, v in tok(question, return_tensors="pt").items()
    }
    resolved, user_override = resolve_max_new_tokens(
        options, device, default=32
    )
    max_tokens = (
        resolved if user_override else cap_max_new_tokens(resolved, device)
    )

    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
        )

    txt = tok.decode(out[0], skip_special_tokens=True)
    return {"answer": txt, "arch": "qwen_vl"}


def predict_minicpm(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: Any,
) -> Dict[str, Any]:
    """Predict using MiniCPM-V model."""
    if model is None:
        return {}
    tok = tokenizer or getattr(model, "tokenizer", None)
    if tok is None:
        return {}

    chat = getattr(model, "chat", None)
    msgs = [{"role": "user", "content": question}]
    resolved, user_override = resolve_max_new_tokens(
        options, device, default=32
    )
    max_tokens = (
        resolved if user_override else cap_max_new_tokens(resolved, device)
    )

    if callable(chat):
        try:
            out = chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tok,
                sampling=False,
                max_new_tokens=max_tokens,
            )
            ans = out[0] if isinstance(out, (list, tuple)) and out else out
            return {"answer": str(ans), "arch": "minicpm_vlm_chat"}
        except Exception as e:
            log.error("minicpm chat failed: %s", e)

    try:
        ans = _minicpm_manual_decode(question, max_tokens, model, tok, device)
        return {"answer": ans, "arch": "minicpm_vlm_manual"}
    except Exception as e:
        log.error("minicpm manual decode failed: %s", e)
        return {}


def predict_vlm(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    model_id: str,
) -> Dict[str, Any]:
    """Predict using generic VLM model."""
    from .gemma import build_vlm_inputs

    enc = build_vlm_inputs(
        image, question, options, processor, model, device, model_id
    )

    if enc.get("_pipeline_text"):
        return {"answer": enc["_pipeline_text"], "arch": "vlm_pipeline"}

    if enc.get("_skip_generation") or model is None:
        log.info(
            "vlm: skip_generation=%s model_none=%s",
            enc.get("_skip_generation"),
            model is None,
        )
        return {}

    strip_processor_only_kwargs(enc, model_id)

    try:
        log.info("vlm: starting generate with keys=%s", list(enc.keys()))
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )
        log.info("vlm: generate completed")
    except Exception as e:
        log.error("vlm: generate failed: %s", e)
        return {}

    tokenizer = get_tokenizer(processor=processor, model=model)
    answer = decode_output(out, processor, tokenizer)
    return {"answer": answer, "arch": "vlm"}


def predict_generic_vqa(
    image: Any,
    question: str,
    options: Dict[str, Any],
    pipe: Any,
) -> Dict[str, Any]:
    """Predict using VQA pipeline."""
    out = pipe(image=image, question=question)
    first = out[0] if isinstance(out, list) else out
    ans = first.get("generated_text") or first.get("answer")
    return {"answer": ans, "arch": "generic_vqa"} if ans else {}


def predict_yi_vl(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    arch: str,
) -> Dict[str, Any]:
    """Predict using Yi-VL model."""
    if arch == "yi_vl_unsupported" or model is None:
        log.warning("yi_vl: model unsupported or not loaded")
        return {}

    try:
        tokenizer = get_tokenizer(processor=processor, model=model)
        question = ensure_image_tokens(question, 1, tokenizer)
        enc = processor(text=question, images=[image], return_tensors="pt")
        enc = move_to_device(enc, device)
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )

        answer = decode_output(out, processor, tokenizer)
        return {"answer": answer, "arch": "yi_vl"}
    except Exception as e:
        log.error("yi_vl predict failed: %s", e)
        return {}


def predict_internvl(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    arch: str,
) -> Dict[str, Any]:
    """Predict using InternVL2 model."""
    if arch == "internvl_unsupported" or model is None:
        log.warning("internvl: model unsupported or not loaded")
        return {}

    # Try chat-style interface first
    chat = getattr(model, "chat", None)
    if callable(chat):
        try:
            resp = chat(image=image, question=question)
            return {"answer": str(resp), "arch": "internvl_chat"}
        except Exception as e:
            log.info("internvl chat failed: %s", e)

    try:
        tokenizer = get_tokenizer(processor=processor, model=model)
        question = ensure_image_tokens(question, 1, tokenizer)
        enc = processor(text=question, images=[image], return_tensors="pt")
        enc = move_to_device(enc, device)
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )

        answer = decode_output(out, processor, tokenizer)
        return {"answer": answer, "arch": "internvl"}
    except Exception as e:
        log.error("internvl predict failed: %s", e)
        return {}


def predict_kosmos2(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    arch: str,
    model_id: str,
) -> Dict[str, Any]:
    """Predict using Kosmos-2 model."""
    if arch == "kosmos2_unsupported" or model is None:
        log.warning("kosmos2: model unsupported or not loaded")
        return {}

    try:
        tokenizer = get_tokenizer(processor=processor, model=model)
        question = ensure_image_tokens(question, 1, tokenizer)
        enc = processor(text=question, images=[image], return_tensors="pt")
        enc = move_to_device(enc, device)
        strip_processor_only_kwargs(enc, model_id)
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )

        answer = decode_output(out, processor, tokenizer)
        return {"answer": answer, "arch": "kosmos2"}
    except Exception as e:
        log.error("kosmos2 predict failed: %s", e)
        return {}


def predict_florence2(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    arch: str,
    model_id: str,
) -> Dict[str, Any]:
    """Predict using Florence-2 model."""
    if arch == "florence2_unsupported" or model is None:
        log.warning("florence2: model unsupported or not loaded")
        return {}

    try:
        # Florence-2 uses task prompt tokens for VQA
        task_prompt = f"<VQA> {question}"
        enc = processor(text=task_prompt, images=[image], return_tensors="pt")
        enc = move_to_device(enc, device)
        strip_processor_only_kwargs(enc, model_id)
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )

        tokenizer = get_tokenizer(processor=processor, model=model)
        answer = decode_output(out, processor, tokenizer)
        return {"answer": answer, "arch": "florence2"}
    except Exception as e:
        log.error("florence2 predict failed: %s", e)
        return {}


def predict_cogvlm(
    image: Any,
    question: str,
    options: Dict[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    arch: str,
) -> Dict[str, Any]:
    """Predict using CogVLM2 model."""
    if arch == "cogvlm_unsupported" or model is None:
        log.warning("cogvlm: model unsupported or not loaded")
        return {}

    # Try chat-style interface first
    chat = getattr(model, "chat", None)
    if callable(chat):
        try:
            resp = chat(image=image, query=question)
            return {"answer": str(resp), "arch": "cogvlm_chat"}
        except Exception as e:
            log.info("cogvlm chat failed: %s", e)

    try:
        tokenizer = get_tokenizer(processor=processor, model=model)
        question = ensure_image_tokens(question, 1, tokenizer)
        enc = processor(text=question, images=[image], return_tensors="pt")
        enc = move_to_device(enc, device)
        resolved, user_override = resolve_max_new_tokens(
            options, device, default=32
        )
        max_tokens = (
            resolved if user_override else cap_max_new_tokens(resolved, device)
        )

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1
            )

        answer = decode_output(out, processor, tokenizer)
        return {"answer": answer, "arch": "cogvlm"}
    except Exception as e:
        log.error("cogvlm predict failed: %s", e)
        return {}


def _minicpm_manual_decode(
    question: str,
    max_len: int,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> str:
    """Manual decoding for MiniCPM-V when chat API is unavailable."""
    tok = require(tokenizer, "MiniCPM-V manual decode missing tokenizer")
    llm = getattr(model, "llm", None) or require(
        model, "MiniCPM-V manual decode missing model"
    )

    if hasattr(llm, "parameters"):
        dev = next(llm.parameters()).device
    else:
        dev = device or torch.device("cpu")

    enc = {
        k: (v.to(dev) if hasattr(v, "to") else v)
        for k, v in tok(question, return_tensors="pt").items()
    }

    if hasattr(llm, "generate"):
        try:
            out = llm.generate(
                **enc, max_new_tokens=max_len, do_sample=False, num_beams=1
            )
            text = tok.decode(out[0], skip_special_tokens=True).strip()
            if text:
                return text
        except Exception as e:
            log.debug("minicpm manual generate failed: %s", e)

    out = llm(**enc)
    logits = getattr(
        out, "logits", out[0] if isinstance(out, (list, tuple)) else None
    )
    if logits is None:
        raise RuntimeError("MiniCPM-V manual decode produced no logits")

    token_id = int(logits[0, -1].argmax().item())
    text = tok.decode([token_id], skip_special_tokens=True).strip()
    return text or (tok.decode([token_id]) or "")
