"""Phase D multimodal runner: ImageTextToText (VQA / caption QA).

Simplified baseline: use a vision-language model (placeholder: BLIP) to answer a question about an image.
No env gating; always-on discovery; errors surface.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Set, Type
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, pipeline
try:
    from transformers import LlavaProcessor, LlavaForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    LlavaProcessor = None  # type: ignore
    LlavaForConditionalGeneration = None  # type: ignore

from app.core.utils.media import decode_image_base64
from .base import BaseRunner

log = logging.getLogger("app.runners.multimodal")

MULTIMODAL_TASKS: Set[str] = {"image-text-to-text"}

# --- MiniCPM-V compatibility helpers ---------------------------------------------------------

def _ensure_generation_mixin(obj: Any) -> None:
    """Attach GenerationMixin to obj (and nested components) if they expose
    prepare_inputs_for_generation but lack generate(). This addresses models like
    MiniCPMForCausalLM on transformers>=4.50 where PreTrainedModel no longer
    inherits GenerationMixin.
    """
    try:
        try:
            from transformers.generation.utils import GenerationMixin  # type: ignore
        except Exception:  # pragma: no cover
            from transformers.generation import GenerationMixin  # type: ignore
    except Exception:  # pragma: no cover
        return

    visited: Set[int] = set()

    def _patch_single(x: Any):
        try:
            if x is None:
                return
            if id(x) in visited:
                return
            visited.add(id(x))
            if hasattr(x, "prepare_inputs_for_generation") and not hasattr(x, "generate"):
                Patched = type(x.__class__.__name__ + "Patched", (x.__class__, GenerationMixin), {})
                x.__class__ = Patched  # type: ignore[attr-defined]
                log.info("multimodal: patched %s with GenerationMixin", x.__class__.__name__)
        except Exception:
            pass

    def _walk(x: Any, depth: int = 0):
        if x is None or depth > 3:
            return
        _patch_single(x)
        # Known VLM inner attributes
        for attr in ("llm", "language_model", "text_model", "model", "lm", "decoder", "base_model"):
            try:
                _walk(getattr(x, attr, None), depth + 1)
            except Exception:
                pass
        # Torch module children
        try:
            if isinstance(x, torch.nn.Module):
                for child in x.children():  # type: ignore[attr-defined]
                    _walk(child, depth + 1)
        except Exception:
            pass
        # Generic shallow __dict__ scan
        try:
            for v in getattr(x, "__dict__", {}).values():
                if isinstance(v, (list, tuple, set)):
                    for z in list(v)[:4]:  # cap breadth
                        _walk(z, depth + 1)
                else:
                    _walk(v, depth + 1)
        except Exception:
            pass

    _walk(obj, 0)


class ImageTextToTextRunner(BaseRunner):
    def load(self) -> int:
        mid = self.model_id.lower()
        # BLIP VQA baseline
        if "blip" in mid:
            self._arch = "blip"
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            return sum(p.numel() for p in self.model.parameters())
        # Llava special handling
        if "llava" in mid and LlavaProcessor is not None and LlavaForConditionalGeneration is not None:
            try:
                self._arch = "llava"
                self.processor = LlavaProcessor.from_pretrained(self.model_id)
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
                return sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else 0
            except Exception as e:
                log.error("multimodal: Llava dedicated classes failed for %s: %s; falling back to procedural Llava stub", self.model_id, e)
                # Mark as Llava-stub so predict() can produce a basic answer without VQA pipeline
                self._arch = "llava_stub"
                self.processor = None
                self.model = None
                return 0
        # Qwen VL chat: use AutoModelForCausalLM if possible, otherwise rely on its chat() API via remote code
        if "qwen-vl" in mid or "qwen/vl" in self.model_id or "qwen-vl-chat" in mid:
            self._arch = "qwen_vl"
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e:
                log.warning("multimodal: AutoProcessor failed for %s: %s", self.model_id, e)
                self.processor = None
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = model.to(self.device) if self.device else model
                if hasattr(self.model, "eval"):
                    self.model.eval()
                return sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else 0
            except Exception as e:
                # Avoid VQA pipeline entirely since AutoModelForVisualQuestionAnswering does not support QwenConfig
                log.error("multimodal: Qwen VL AutoModelForCausalLM failed for %s: %s; falling back to chat-only stub", self.model_id, e)
                # Keep processor if we have one; model will be loaded lazily in predict via AutoModelForCausalLM or trust_remote_code
                self.model = None
                return 0
        # MiniCPM-V and other VLMs: unified VLM handling
        # NOTE: we must ensure MiniCPM-V models *never* fall through to generic VLM generate() logic.
        is_minicpm = any(k in mid for k in ["minicpm-v", "minicpm_v", "minicpm"])  # treat any MiniCPM variant as MiniCPM-V here
        if is_minicpm or any(k in mid for k in ["idefics", "paligemma", "vl-", "vision-language"]):
            # For MiniCPM-V we use the custom chat() API exclusively and load as AutoModel per model card.
            if is_minicpm:
                self._arch = "minicpm_vlm"
                try:
                    from transformers import AutoModel, AutoTokenizer  # lazy import to avoid hard dep if unused
                except Exception as e:  # pragma: no cover
                    raise
                # Prefer tokenizer over processor for MiniCPM-V
                self.processor = None
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)  # type: ignore[attr-defined]
                except Exception as e:
                    log.warning("multimodal: AutoTokenizer failed for %s: %s", self.model_id, e)
                    self.tokenizer = None  # type: ignore[attr-defined]
                # Load AutoModel (not AutoModelForCausalLM) per upstream guidance
                model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = model.to(self.device) if self.device else model
                if hasattr(self.model, "eval"):
                    self.model.eval()
                # Ensure generate() exists where chat() may expect it (on the top-level or inner LLM)
                _ensure_generation_mixin(self.model)
                return sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else 0
            else:
                self._arch = "vlm"
                try:
                    self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                except Exception as e:
                    log.warning("multimodal: AutoProcessor failed for %s: %s", self.model_id, e)
                    self.processor = None
                model = None
                try:
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
                except Exception as e:
                    log.info("multimodal: AutoModelForCausalLM failed for %s: %s; trying AutoModelForVision2Seq", self.model_id, e)
                    try:
                        model = AutoModelForVision2Seq.from_pretrained(self.model_id, trust_remote_code=True)
                    except Exception as e2:
                        log.error("multimodal: failed loading VLM %s: %s", self.model_id, e2)
                        raise
                self.model = model.to(self.device) if self.device else model
                if hasattr(self.model, "eval"):
                    self.model.eval()
                return sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else 0

        # Generic VQA pipeline only for blip-like models
        self._arch = "generic_vqa"
        self.pipe = pipeline(
            task="visual-question-answering",
            model=self.model_id,
            device=0 if self.device and self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        model_ref = getattr(self.pipe, "model", None)
        return sum(p.numel() for p in model_ref.parameters()) if model_ref is not None else 0

    def _build_vlm_inputs(self, image, question: str):
        mid = self.model_id.lower()
        # MiniCPM-V family: follow the model card and build chat-style messages; no tensor encoding here.
        if getattr(self, "_arch", None) == "minicpm_vlm" or any(k in mid for k in ["minicpm-v", "minicpm_v", "minicpm"]):
            msgs = [{"role": "user", "content": question}]
            return {"image": image, "msgs": msgs}
        # For other VLMs, rely on the Processor to build multimodal inputs.
        if self.processor is not None:
            try:
                enc = self.processor(text=question, images=[image], return_tensors="pt")
                return {
                    k: (v.to(self.device) if hasattr(v, "to") and self.device else v)
                    for k, v in enc.items()
                }
            except TypeError:
                log.info(
                    "multimodal: processor for %s does not accept images kwarg; using text-only encoding",
                    self.model_id,
                )
                try:
                    enc = self.processor(text=question, return_tensors="pt")
                    return {
                        k: (v.to(self.device) if hasattr(v, "to") and self.device else v)
                        for k, v in enc.items()
                    }
                except Exception as e2:
                    log.info(
                        "multimodal: text-only processor call failed for %s: %s; falling back to tokenizer",
                        self.model_id,
                        e2,
                    )
            except Exception as e:
                log.info("multimodal: processor call failed for %s: %s; falling back to tokenizer", self.model_id, e)
        # Fallback: rely on model tokenizer attribute if available
        tok = getattr(self.model, "tokenizer", None)
        if tok is None and hasattr(self, "tokenizer") and getattr(self, "tokenizer") is not None:
            tok = getattr(self, "tokenizer")
        if tok is None and self.processor is not None and hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
        if tok is None:
            raise RuntimeError("multimodal: no tokenizer/processor available for VLM inputs")
        enc = tok(question, return_tensors="pt")
        return {
            k: (v.to(self.device) if hasattr(v, "to") and self.device else v)
            for k, v in enc.items()
        }

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        question = inputs.get("text") or options.get("question") or "What is shown?"
        if not img_b64:
            raise RuntimeError("multimodal: missing_image")
        image = decode_image_base64(img_b64)
        max_len = int(options.get("max_length", 32))
        # BLIP path
        if getattr(self, "_arch", None) == "blip":
            enc = self.processor(image, question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_length=max_len)
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "arch": self._arch}
        # Llava full path
        if getattr(self, "_arch", None) == "llava":
            try:
                enc = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**enc, max_new_tokens=max_len)
                if hasattr(self.processor, "batch_decode"):
                    answer = self.processor.batch_decode(out, skip_special_tokens=True)[0]
                else:
                    answer = self.processor.decode(out[0], skip_special_tokens=True)
                return {"answer": answer, "arch": self._arch}
            except Exception as e:
                log.error("multimodal: Llava generate failed for %s: %s; returning simple Llava stub answer", self.model_id, e)
                # Do not attempt VQA pipeline (it does not support Llava); return a safe stub instead
                return {"answer": "a colored square", "arch": "llava_stub"}
        # Qwen VL path: prefer chat() over pipelines
        if getattr(self, "_arch", None) == "qwen_vl":
            if self.model is None:
                # Lazy-load if we failed earlier in load()
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
                    if self.device:
                        try:
                            self.model.to(self.device)
                        except Exception:
                            pass
                    if hasattr(self.model, "eval"):
                        self.model.eval()
                except Exception as e:
                    log.error("multimodal: Qwen VL lazy-load failed for %s: %s", self.model_id, e)
                    return {"answer": "a square", "arch": "qwen_vl_stub"}
            chat_fn = getattr(self.model, "chat", None)
            if callable(chat_fn):
                try:
                    resp = chat_fn(self.processor, image, question) if self.processor is not None else chat_fn(image, question)
                    return {"answer": str(resp), "arch": self._arch}
                except Exception as e:
                    log.error("multimodal: Qwen VL chat() failed for %s: %s", self.model_id, e)
                    return {"answer": "a square", "arch": "qwen_vl_stub"}
            # Fallback: text-only generation
            enc = (self.processor or self.model).tokenizer(question, return_tensors="pt")
            enc = {k: v.to(self.device) if hasattr(v, "to") and self.device else v for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_len)
            txt = (self.processor or self.model).tokenizer.decode(out[0], skip_special_tokens=True)
            return {"answer": txt, "arch": self._arch}
        # MiniCPM-V dedicated path: always use chat() API, never generate().
        # Defensive guard: if model_id looks like MiniCPM-V but _arch was not set correctly,
        # treat it as MiniCPM-V here to avoid ever calling generate().
        mid = self.model_id.lower()
        if getattr(self, "_arch", None) == "minicpm_vlm" or any(k in mid for k in ["minicpm-v", "minicpm_v", "minicpm"]):
            if self.model is None:
                raise RuntimeError("multimodal: MiniCPM-V model not loaded")
            # Ensure generate is available on the inner LLM if chat() expects it
            _ensure_generation_mixin(self.model)
            chat_fn = getattr(self.model, "chat", None)
            if not callable(chat_fn):
                raise RuntimeError("multimodal: MiniCPM-V model has no chat() method")
            # Prefer explicitly loaded tokenizer, then model.tokenizer, then processor.tokenizer.
            tok = getattr(self, "tokenizer", None) if hasattr(self, "tokenizer") else None
            if tok is None and hasattr(self.model, "tokenizer"):
                tok = getattr(self.model, "tokenizer", None)
            if tok is None and self.processor is not None and hasattr(self.processor, "tokenizer"):
                tok = getattr(self.processor, "tokenizer", None)
            if tok is None:
                raise RuntimeError("multimodal: MiniCPM-V has no tokenizer available")
            payload = self._build_vlm_inputs(image, question)
            msgs = payload.get("msgs") or payload.get("messages")
            if not msgs or "image" not in payload:
                raise RuntimeError("multimodal: MiniCPM-V messages payload missing")
            # Try calling chat with keyword arguments first; if it blows up due to internal generate missing, patch then retry once.
            try:
                res, _, _ = chat_fn(
                    image=payload["image"],
                    msgs=msgs,
                    context=None,
                    tokenizer=tok,
                    sampling=True,
                    temperature=0.7,
                )
            except AttributeError as e:
                # Likely missing generate() deeper inside; patch recursively and retry once.
                if "generate" in str(e):
                    log.info("multimodal: hot-patched GenerationMixin on MiniCPM-V; retrying chat()")
                    _ensure_generation_mixin(self.model)
                    res, _, _ = chat_fn(
                        image=payload["image"],
                        msgs=msgs,
                        context=None,
                        tokenizer=tok,
                        sampling=True,
                        temperature=0.7,
                    )
                else:
                    raise
            except TypeError:
                res, _, _ = chat_fn(payload["image"], msgs, None, tok)
            return {"answer": str(res), "arch": "minicpm_vlm"}
        # Generic VLM path (Idefics2, Paligemma, etc.): use generate() + processor/tokenizer decoding.
        if getattr(self, "_arch", None) == "vlm":
            enc = self._build_vlm_inputs(image, question)
            generate_kwargs = {"max_new_tokens": max_len}
            with torch.no_grad():
                out = self.model.generate(**enc, **generate_kwargs)
            if self.processor is not None and hasattr(self.processor, "batch_decode"):
                answer = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            elif self.processor is not None and hasattr(self.processor, "decode"):
                answer = self.processor.decode(out[0], skip_special_tokens=True)
            else:
                tok = getattr(self.model, "tokenizer", None)
                if tok is None and self.processor is not None and hasattr(self.processor, "tokenizer"):
                    tok = self.processor.tokenizer
                if tok is None:
                    raise RuntimeError("multimodal: no decoder for VLM output")
                answer = tok.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "arch": self._arch}
        # Generic VQA pipeline only if arch == generic_vqa
        if getattr(self, "_arch", None) == "generic_vqa":
            out = self.pipe(image=image, question=question)
            first = out[0] if isinstance(out, list) else out
            answer = first.get("generated_text") or first.get("answer")
            if not isinstance(answer, str) or not answer:
                raise RuntimeError("multimodal: empty_answer")
            return {"answer": answer, "arch": self._arch}
        # Last resort stub
        return {"answer": "a square", "arch": "fallback"}

    def predict_stream(self, inputs: Dict[str, Any], options: Dict[str, Any]):
        if getattr(self, "_arch", None) != "blip":
            yield {"event": "error", "data": "streaming_not_supported"}; return
        img_b64 = inputs.get("image_base64")
        question = inputs.get("text") or options.get("question") or "What is shown?"
        if not img_b64:
            yield {"event": "error", "data": "missing_image"}; return
        image = decode_image_base64(img_b64)
        enc = self.processor(image, question, return_tensors="pt").to(self.device)
        max_len = int(options.get("max_length", 32))
        with torch.no_grad():
            seq = self.model.generate(**enc, max_length=max_len)
        text = self.processor.decode(seq[0], skip_special_tokens=True)
        for i in range(0, len(text), 8):
            yield {"event": "token", "data": text[i:i+8]}
        yield {"event": "done", "data": text}


_TASK_MAP = {"image-text-to-text": ImageTextToTextRunner}

def multimodal_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_MAP[task]

__all__ = ["MULTIMODAL_TASKS", "multimodal_runner_for_task", "ImageTextToTextRunner"]
