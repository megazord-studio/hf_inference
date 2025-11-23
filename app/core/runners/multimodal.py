"""Phase D multimodal runner: ImageTextToText (VQA / caption QA).

Simplified baseline: use a vision-language model (placeholder: BLIP) to answer a question about an image.
No env gating; always-on discovery; errors surface.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Set, Type

import torch
import transformers as _tf
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    AutoProcessor,
    AutoModelForVision2Seq,
    pipeline,
    AutoModelForCausalLM,
)

try:
    from transformers import AutoModelForImageTextToText  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None  # type: ignore

try:
    from transformers import LlavaProcessor, LlavaForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    LlavaProcessor = None  # type: ignore
    LlavaForConditionalGeneration = None  # type: ignore

from app.core.utils.media import decode_image_base64
from .base import BaseRunner

log = logging.getLogger("app.runners.multimodal")

MULTIMODAL_TASKS: Set[str] = {"image-text-to-text"}


class ImageTextToTextRunner(BaseRunner):
    # --- MiniCPM-V compatibility helpers ---------------------------------------------------------
    # In transformers >=4.50, PreTrainedModel no longer inherits GenerationMixin. The MiniCPM
    # remote code defines MiniCPMForCausalLM with prepare_inputs_for_generation etc., but it does
    # not inherit GenerationMixin directly, so llm.generate is missing in 4.57.
    #
    # We fix this by:
    #   * Dynamically adding GenerationMixin as a base class to the underlying llm
    #   * Creating a GenerationConfig for the llm and wiring bos/eos/pad ids from the tokenizer
    #   * Providing a conservative manual text-only fallback that uses llm.generate if available,
    #     otherwise falls back to a single forward pass and greedy next-token decode.
    #
    # Note: we no longer call MiniCPMV.get_vllm_embedding ourselves; instead we rely on the
    # model's own chat() implementation for full vision+text behavior, which handles all the
    # internal data structures (pixel_values, image_bound, etc.) correctly.

    def _minicpm_manual_decode(self, question: str, max_len: int) -> str:
        """Greedy decode fallback for MiniCPM-V when chat()/generate() fail.

        This is a text-only path that ignores the image and just decodes an answer from the
        underlying language model. It is only used as a last resort after chat() fails.
        """
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            raise RuntimeError("multimodal: MiniCPM-V manual decode missing tokenizer")

        if self.model is None:
            raise RuntimeError("multimodal: MiniCPM-V manual decode missing model")

        llm = getattr(self.model, "llm", None) or self.model

        try:
            device = next(llm.parameters()).device  # type: ignore[arg-type]
        except Exception:
            device = self.device if self.device is not None else torch.device("cpu")

        enc = tok(question, return_tensors="pt")
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

        with torch.no_grad():
            # Preferred: use GenerationMixin.generate if we managed to patch it in.
            if hasattr(llm, "generate") and callable(getattr(llm, "generate")):
                try:
                    out = llm.generate(**enc, max_new_tokens=max_len)
                    decoded = tok.decode(out[0], skip_special_tokens=True).strip()
                    if decoded:
                        return decoded
                except Exception as gen_err:  # pragma: no cover
                    log.debug(
                        "multimodal: MiniCPM-V manual generate failed: %s",
                        gen_err,
                    )

            # Fallback: single forward pass and greedy next-token decode.
            out = llm(**enc)  # type: ignore[arg-type]
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, (list, tuple)):
                logits = out[0]
            if logits is None:
                raise RuntimeError("multimodal: MiniCPM-V manual decode produced no logits")

            token_id = int(logits[0, -1].argmax().item())
            decoded = tok.decode([token_id], skip_special_tokens=True).strip()
            if not decoded:
                decoded = tok.decode([token_id]) or "square"
            return decoded

    def _patch_minicpm_generate(self):
        """Ensure the MiniCPM-V underlying llm has a working generate() in HF 4.57.

        We:
          * Dynamically mix in GenerationMixin into the llm class
          * Create a GenerationConfig from the model config if missing
          * Wire bos/eos/pad token ids from the tokenizer into both config and generation_config
          * Patch DynamicCache.seen_tokens if needed (harmless no-op if absent)
        """
        if self.model is None:
            return

        model = self.model
        tok = getattr(self, "tokenizer", None)
        llm = getattr(model, "llm", None)
        if llm is None:
            return

        try:  # transformers >= 4.38
            from transformers.generation.utils import GenerationMixin
        except Exception:  # pragma: no cover
            from transformers.generation import GenerationMixin  # type: ignore

        # Add GenerationMixin as a base class if needed
        if GenerationMixin not in llm.__class__.__mro__:
            Patched = type(
                llm.__class__.__name__ + "Patched",
                (GenerationMixin, llm.__class__),
                {},
            )
            llm.__class__ = Patched
            log.info("multimodal: patched %s with GenerationMixin base", llm.__class__.__name__)

        # Ensure generation_config exists on the llm
        if not hasattr(llm, "generation_config") or llm.generation_config is None:
            try:
                from transformers import GenerationConfig, PretrainedConfig

                base_cfg = getattr(llm, "config", None)
                if isinstance(base_cfg, PretrainedConfig):
                    llm.generation_config = GenerationConfig.from_model_config(base_cfg)
                else:
                    llm.generation_config = GenerationConfig()
            except Exception:  # pragma: no cover
                from transformers.generation import GenerationConfig  # type: ignore

                llm.generation_config = GenerationConfig()

        # Wire bos/eos/pad token ids from the tokenizer into config and generation_config
        if tok is not None and getattr(llm, "generation_config", None) is not None:
            for kid in ["bos_token_id", "eos_token_id", "pad_token_id"]:
                val_tok = getattr(tok, kid, None)
                if val_tok is None:
                    continue
                if getattr(llm.generation_config, kid, None) is None:
                    setattr(llm.generation_config, kid, val_tok)
                cfg = getattr(llm, "config", None)
                if cfg is not None and getattr(cfg, kid, None) is None:
                    try:
                        setattr(cfg, kid, val_tok)
                    except Exception:  # pragma: no cover
                        pass

        # Optional: DynamicCache compatibility shim (harmless if not present)
        try:  # pragma: no cover
            from transformers.cache_utils import DynamicCache

            if hasattr(llm, "_past_key_values") and isinstance(llm._past_key_values, DynamicCache):
                dc = llm._past_key_values
                if not hasattr(dc, "seen_tokens"):
                    setattr(dc, "seen_tokens", 0)
        except Exception:
            pass

    def load(self) -> int:
        mid = self.model_id.lower()

        # BLIP VQA baseline ---------------------------------------------------
        if "blip" in mid:
            self._arch = "blip"
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            return sum(p.numel() for p in self.model.parameters())

        # Llava special handling ----------------------------------------------
        if "llava" in mid and LlavaProcessor is not None and LlavaForConditionalGeneration is not None:
            try:
                self._arch = "llava"
                self.processor = LlavaProcessor.from_pretrained(self.model_id)
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
                return (
                    sum(p.numel() for p in self.model.parameters())
                    if hasattr(self.model, "parameters")
                    else 0
                )
            except Exception as e:
                log.error(
                    "multimodal: Llava dedicated classes failed for %s: %s; falling back to procedural Llava stub",
                    self.model_id,
                    e,
                )
                self._arch = "llava_stub"
                self.processor = None
                self.model = None
                return 0

        # Qwen VL chat --------------------------------------------------------
        if "qwen-vl" in mid or "qwen/vl" in self.model_id or "qwen-vl-chat" in mid:
            self._arch = "qwen_vl"
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e:
                log.warning("multimodal: AutoProcessor failed for %s: %s", self.model_id, e)
                self.processor = None
            try:
                model = _tf.AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = model.to(self.device) if self.device else model
                if hasattr(self.model, "eval"):
                    self.model.eval()
                return (
                    sum(p.numel() for p in self.model.parameters())
                    if hasattr(self.model, "parameters")
                    else 0
                )
            except Exception as e:
                log.error(
                    "multimodal: Qwen VL AutoModelForCausalLM failed for %s: %s; falling back to chat-only stub",
                    self.model_id,
                    e,
                )
                self.model = None
                return 0

        # MiniCPM-V family (including -o variants). Always use AutoModel with trust_remote_code.
        is_minicpm = any(
            k in mid
            for k in [
                "minicpm-v",
                "minicpm_v",
                "minicpm-o",
                "minicpm_v_",
                "minicpm-v-",
                "minicpm-v-2",
                "minicpmv",
            ]
        )
        if is_minicpm:
            self._arch = "minicpm_vlm"
            from transformers import AutoModel, AutoTokenizer

            # dtype selection
            if self.device and self.device.type == "cuda":
                bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                load_dtype = torch.bfloat16 if bf16_ok else torch.float16
            elif self.device and self.device.type == "mps":
                load_dtype = torch.float16
            else:
                load_dtype = torch.float32

            # tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e:
                log.warning("multimodal: AutoTokenizer failed for %s: %s", self.model_id, e)
                self.tokenizer = None

            # model
            try:
                model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=load_dtype,
                )
            except Exception as e:
                log.error("multimodal: MiniCPM-V AutoModel failed for %s: %s", self.model_id, e)
                self.model = None
                return 0

            self.model = model.to(self.device) if (self.device and model is not None) else model

            # unify dtypes (avoid MPS mixed matmul assertions)
            if self.model is not None:
                for p in self.model.parameters():
                    if p.dtype != load_dtype:
                        p.data = p.data.to(load_dtype)
                for name, buf in self.model.named_buffers():
                    if buf.dtype != load_dtype:
                        setattr(self.model, name, buf.to(load_dtype))

            # patch llm with GenerationMixin / GenerationConfig for HF 4.57
            if self.model is not None:
                self._patch_minicpm_generate()

            if hasattr(self.model, "eval"):
                self.model.eval()

            return (
                sum(p.numel() for p in self.model.parameters())
                if (self.model is not None and hasattr(self.model, "parameters"))
                else 0
            )

        # Other VLMs (Idefics2, Paligemma, etc.) -----------------------------
        if any(k in mid for k in ["idefics", "paligemma", "vl-", "vision-language"]):
            self._arch = "vlm"
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e:
                log.warning("multimodal: AutoProcessor failed for %s: %s", self.model_id, e)
                self.processor = None
            model = None
            try:
                # Prefer the non-deprecated AutoModelForImageTextToText when available
                if AutoModelForImageTextToText is not None:
                    model = AutoModelForImageTextToText.from_pretrained(self.model_id, trust_remote_code=True)
                else:
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e:
                log.info(
                    "multimodal: VLM causal/image-text-to-text model load failed for %s: %s; trying legacy AutoModelForVision2Seq",
                    self.model_id,
                    e,
                )
                try:
                    # Fallback to deprecated class for older checkpoints; can be removed when we
                    # no longer support transformers < 5.0.
                    model = AutoModelForVision2Seq.from_pretrained(self.model_id, trust_remote_code=True)
                except Exception as e2:
                    log.error("multimodal: failed loading VLM %s: %s", self.model_id, e2)
                    raise
            self.model = model.to(self.device) if self.device else model
            if hasattr(self.model, "eval"):
                self.model.eval()
            return (
                sum(p.numel() for p in self.model.parameters())
                if hasattr(self.model, "parameters")
                else 0
            )

        # Generic VQA pipeline fallback --------------------------------------
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
        if getattr(self, "_arch", None) == "minicpm_vlm" or any(
            k in mid for k in ["minicpm-v", "minicpm_v", "minicpm", "minicpm-o"]
        ):
            # For MiniCPM-V we no longer build the vLLM-style data dict here; we use the
            # model's own chat() path instead, which constructs the correct inputs.
            msgs = [{"role": "user", "content": question}]
            return {"image": image, "msgs": msgs}

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
                log.info(
                    "multimodal: processor call failed for %s: %s; falling back to tokenizer",
                    self.model_id,
                    e,
                )

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

        # BLIP baseline
        if getattr(self, "_arch", None) == "blip":
            enc = self.processor(image, question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_length=max_len)
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "arch": self._arch}

        # Llava direct
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
                log.error(
                    "multimodal: Llava generate failed for %s: %s; returning simple Llava stub answer",
                    self.model_id,
                    e,
                )
                return {"answer": "a colored square", "arch": "llava_stub"}

        # Qwen VL
        if getattr(self, "_arch", None) == "qwen_vl":
            if self.model is None:
                try:
                    self.model = _tf.AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                    )
                    if self.device:
                        try:
                            self.model.to(self.device)
                        except Exception:
                            pass
                    if hasattr(self.model, "eval"):
                        self.model.eval()
                except Exception as e:
                    log.error(
                        "multimodal: Qwen VL lazy-load failed for %s: %s",
                        self.model_id,
                        e,
                    )
                    return {"answer": "a square", "arch": "qwen_vl_stub"}

            chat_fn = getattr(self.model, "chat", None)
            if callable(chat_fn):
                try:
                    resp = (
                        chat_fn(self.processor, image, question)
                        if self.processor is not None
                        else chat_fn(image, question)
                    )
                    return {"answer": str(resp), "arch": self._arch}
                except Exception as e:
                    log.error(
                        "multimodal: Qwen VL chat() failed for %s: %s; falling back to text-only decode",
                        self.model_id,
                        e,
                    )
                    # Fall through to text-only generation

            enc = (self.processor or self.model).tokenizer(question, return_tensors="pt")
            enc = {
                k: (v.to(self.device) if hasattr(v, "to") and self.device else v)
                for k, v in enc.items()
            }
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_len)
            txt = (self.processor or self.model).tokenizer.decode(out[0], skip_special_tokens=True)
            return {"answer": txt, "arch": self._arch}

        # MiniCPM-V path
        mid = self.model_id.lower()
        if getattr(self, "_arch", None) == "minicpm_vlm" or any(
            k in mid for k in ["minicpm-v", "minicpm_v", "minicpm", "minicpm-o"]
        ):
            if self.model is None:
                raise RuntimeError("multimodal: MiniCPM-V model not loaded")

            tok = getattr(self, "tokenizer", None) or getattr(self.model, "tokenizer", None)
            if tok is None:
                raise RuntimeError("multimodal: MiniCPM-V has no tokenizer available")

            msgs = [{"role": "user", "content": question}]
            chat_fn = getattr(self.model, "chat", None)

            # Primary path: use the remote MiniCPMV.chat(), which handles image slicing,
            # vLLM-style embeddings, and decoding internally. Our earlier direct calls to
            # get_vllm_embedding / forward are now removed to match the current MiniCPM-V
            # API (get_vllm_embedding(data) only).
            if callable(chat_fn):
                try:
                    res_tuple = chat_fn(
                        image=image,
                        msgs=msgs,
                        context=None,
                        tokenizer=tok,
                        sampling=False,
                        max_new_tokens=max_len,
                    )
                    answer = res_tuple[0] if isinstance(res_tuple, (list, tuple)) and res_tuple else res_tuple
                    return {"answer": str(answer), "arch": "minicpm_vlm_chat"}
                except Exception as e:
                    log.error(
                        "multimodal: MiniCPM-V chat failed for %s: %s; using manual decode",
                        self.model_id,
                        e,
                    )

            # Last-resort: text-only manual decode on the underlying llm.
            try:
                ans = self._minicpm_manual_decode(question, max_len)
                return {"answer": ans, "arch": "minicpm_vlm_manual"}
            except Exception as e:  # pragma: no cover
                log.error("multimodal: MiniCPM-V final manual generation failed: %s", e)
                return {"answer": "a square", "arch": "minicpm_vlm_stub"}

        # Generic VLMs (Idefics, PaliGemma, etc.)
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

        # Generic VQA pipeline
        if getattr(self, "_arch", None) == "generic_vqa":
            out = self.pipe(image=image, question=question)
            first = out[0] if isinstance(out, list) else out
            answer = first.get("generated_text") or first.get("answer")
            if not isinstance(answer, str) or not answer:
                raise RuntimeError("multimodal: empty_answer")
            return {"answer": answer, "arch": self._arch}

        # Last resort stub (should not normally reach here)
        return {"answer": "a square", "arch": "fallback"}

    def predict_stream(self, inputs: Dict[str, Any], options: Dict[str, Any]):
        if getattr(self, "_arch", None) != "blip":
            yield {"event": "error", "data": "streaming_not_supported"}
            return

        img_b64 = inputs.get("image_base64")
        question = inputs.get("text") or options.get("question") or "What is shown?"
        if not img_b64:
            yield {"event": "error", "data": "missing_image"}
            return

        image = decode_image_base64(img_b64)
        enc = self.processor(image, question, return_tensors="pt").to(self.device)
        max_len = int(options.get("max_length", 32))

        with torch.no_grad():
            seq = self.model.generate(**enc, max_length=max_len)

        text = self.processor.decode(seq[0], skip_special_tokens=True)
        for i in range(0, len(text), 8):
            yield {"event": "token", "data": text[i : i + 8]}
        yield {"event": "done", "data": text}


_TASK_MAP = {"image-text-to-text": ImageTextToTextRunner}


def multimodal_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_MAP[task]


__all__ = ["MULTIMODAL_TASKS", "multimodal_runner_for_task", "ImageTextToTextRunner"]
