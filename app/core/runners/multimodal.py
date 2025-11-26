"""Phase D multimodal runner: ImageTextToText (VQA / caption QA).

Simplified baseline: use a vision-language model (placeholder: BLIP) to answer a question about an image.
No env gating; always-on discovery; errors surface.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import Type

import torch
import transformers as _tf
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import BlipForQuestionAnswering
from transformers import BlipProcessor
from transformers import LlavaForConditionalGeneration
from transformers import LlavaProcessor
from transformers import pipeline

from app.core.utils.media import decode_image_base64

from .base import BaseRunner

log = logging.getLogger("app.runners.multimodal")

MULTIMODAL_TASKS: Set[str] = {"image-text-to-text"}


class ImageTextToTextRunner(BaseRunner):
    """Image+Text -> Text multimodal runner with clear per-arch flows."""

    def _safe_call(self, fn):
        try:
            return fn()
        except Exception as e:
            log.error("safe_call failed: %s", e)
            return None

    def unload(self) -> None:
        """Release model, processor, tokenizer, pipeline to free memory."""
        for attr in ("model", "processor", "tokenizer", "pipe"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        # Explicitly run garbage collection to free GPU memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------- Public API ---------------------------------
    def load(self) -> int:
        self._arch = self._detect_arch(self.model_id)
        return self._load_by_arch()

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        image = decode_image_base64(self._require(inputs.get("image_base64"), "missing_image"))
        question = inputs.get("text") or options.get("question") or "What is shown?"
        log.info("multimodal.predict start model_id=%s arch=%s", self.model_id, getattr(self, "_arch", None))
        if self._arch == "blip":
            return self._predict_blip(image, question, options)
        if self._arch == "llava":
            return self._predict_llava(image, question, options)
        if self._arch == "qwen_vl":
            return self._predict_qwen_vl(image, question, options)
        if self._arch == "minicpm_vlm":
            return self._predict_minicpm(image, question, options)
        if self._arch == "yi_vl":
            return self._predict_yi_vl(image, question, options)
        if self._arch == "internvl":
            return self._predict_internvl(image, question, options)
        if self._arch == "kosmos2":
            return self._predict_kosmos2(image, question, options)
        if self._arch == "florence2":
            return self._predict_florence2(image, question, options)
        if self._arch == "cogvlm":
            return self._predict_cogvlm(image, question, options)
        if self._arch in {"vlm", "vlm_unsupported"}:
            return self._predict_vlm(image, question, options)
        if self._arch == "generic_vqa":
            return self._predict_generic_vqa(image, question, options)
        return {}

    def predict_stream(self, inputs: Dict[str, Any], options: Dict[str, Any]):
        if getattr(self, "_arch", None) != "blip":
            yield {"event": "error", "data": "streaming_not_supported"}
            return
        image = decode_image_base64(self._require(inputs.get("image_base64"), "missing_image"))
        question = inputs.get("text") or options.get("question") or "What is shown?"
        enc = self.processor(image, question, return_tensors="pt").to(self.device)
        max_len = self._cap_max_new_tokens(int(options.get("max_length", 32)))
        with torch.no_grad():
            seq = self.model.generate(**enc, max_length=max_len, do_sample=False, num_beams=1)
        text = self.processor.decode(seq[0], skip_special_tokens=True)
        for i in range(0, len(text), 8):
            yield {"event": "token", "data": text[i : i + 8]}
        yield {"event": "done", "data": text}

    # ---------------------------- Arch routing --------------------------------
    def _detect_arch(self, model_id: str) -> str:
        mid = model_id.lower()
        if "blip" in mid:
            return "blip"
        if "llava" in mid:
            return "llava"
        if any(k in mid for k in ["qwen-vl", "qwen/vl", "qwen-vl-chat"]):
            return "qwen_vl"
        # Broaden MiniCPM detection to catch IDs like openbmb/MiniCPM-Llama3-V-2_5
        if any(k in mid for k in ["minicpm", "minicpm-v", "minicpm_v", "minicpm-o", "minicpmv"]):
            return "minicpm_vlm"
        # New model architectures
        if "yi-vl" in mid or "yi/vl" in mid:
            return "yi_vl"
        if "internvl" in mid:
            return "internvl"
        if "kosmos-2" in mid or "kosmos2" in mid:
            return "kosmos2"
        if "florence-2" in mid or "florence2" in mid:
            return "florence2"
        if "cogvlm" in mid:
            return "cogvlm"
        if any(k in mid for k in ["idefics", "paligemma", "vl-", "vision-language", "gemma"]):
            return "vlm"
        return "generic_vqa"

    def _load_by_arch(self) -> int:
        if self._arch == "blip":
            return self._load_blip()
        if self._arch == "llava":
            return self._load_llava()
        if self._arch == "qwen_vl":
            return self._load_qwen_vl()
        if self._arch == "minicpm_vlm":
            return self._load_minicpm()
        if self._arch == "yi_vl":
            return self._load_yi_vl()
        if self._arch == "internvl":
            return self._load_internvl()
        if self._arch == "kosmos2":
            return self._load_kosmos2()
        if self._arch == "florence2":
            return self._load_florence2()
        if self._arch == "cogvlm":
            return self._load_cogvlm()
        if self._arch == "vlm":
            return self._load_vlm()
        return self._load_generic_vqa()

    # ----------------------------- Loaders ------------------------------------
    def _load_blip(self) -> int:
        log.info("multimodal: loading BLIP model_id=%s (may download)", self.model_id)
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        return self._count_params(self.model)

    def _load_llava(self) -> int:
        try:
            log.info("multimodal: loading LLaVA model_id=%s (may download)", self.model_id)
            self.processor = LlavaProcessor.from_pretrained(self.model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            return self._count_params(self.model)
        except Exception as e:  # pragma: no cover
            log.error("llava load failed: %s", e)
            self.processor = None
            self.model = None
            return 0

    def _load_qwen_vl(self) -> int:
        log.info("multimodal: loading Qwen-VL model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True))
        model = self._safe_call(lambda: _tf.AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True))
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_minicpm(self) -> int:
        from transformers import AutoModel
        from transformers import AutoTokenizer
        load_dtype = self._select_dtype()
        log.info("multimodal: loading MiniCPM-V model_id=%s (may download)", self.model_id)
        self.tokenizer = self._safe_call(lambda: AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True))
        model = self._safe_call(
            lambda: AutoModel.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=load_dtype)
        )
        self.model = self._to_device(model)
        self._unify_model_dtype(self.model, load_dtype)
        if self.model is not None:
            self._patch_minicpm_generate()
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_vlm(self) -> int:
        log.info("multimodal: loading VLM (auto) model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True))
        model = self._safe_call(lambda: AutoModelForImageTextToText.from_pretrained(self.model_id, trust_remote_code=True))
        if model is None:
            model = self._safe_call(lambda: AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True))
        if model is None:
            model = self._safe_call(lambda: AutoModelForVision2Seq.from_pretrained(self.model_id, trust_remote_code=True))
            if model is None:
                self._arch = "vlm_unsupported"
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_generic_vqa(self) -> int:
        log.info("multimodal: initializing VQA pipeline model_id=%s (may download)", self.model_id)
        self.pipe = pipeline(
            task="visual-question-answering",
            model=self.model_id,
            device=0 if self._is_cuda() else None,
            trust_remote_code=True,
        )
        m = getattr(self.pipe, "model", None)
        return self._count_params(m)

    def _load_yi_vl(self) -> int:
        """Load Yi-VL model. On state_dict size mismatch, log and return 0."""
        log.info("multimodal: loading Yi-VL model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(
            lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        )
        model = self._safe_call(
            lambda: AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        )
        if model is None:
            log.warning("yi_vl: failed to load model, may be state_dict mismatch")
            self._arch = "yi_vl_unsupported"
            return 0
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_internvl(self) -> int:
        """Load InternVL2 model. Uses AutoProcessor + trust_remote_code."""
        log.info("multimodal: loading InternVL model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(
            lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        )
        # Try InternVLChat or AutoModelForCausalLM
        model = self._safe_call(
            lambda: AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        )
        if model is None:
            model = self._safe_call(
                lambda: AutoModelForVision2Seq.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            )
        if model is None:
            log.warning("internvl: failed to load model")
            self._arch = "internvl_unsupported"
            return 0
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_kosmos2(self) -> int:
        """Load Kosmos-2 model. Uses AutoProcessor + appropriate model class."""
        log.info("multimodal: loading Kosmos-2 model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(
            lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        )
        model = self._safe_call(
            lambda: AutoModelForVision2Seq.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        )
        if model is None:
            model = self._safe_call(
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            )
        if model is None:
            log.warning("kosmos2: failed to load model")
            self._arch = "kosmos2_unsupported"
            return 0
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_florence2(self) -> int:
        """Load Florence-2 model using AutoProcessor + appropriate model."""
        log.info("multimodal: loading Florence-2 model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(
            lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        )
        model = self._safe_call(
            lambda: AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        )
        if model is None:
            model = self._safe_call(
                lambda: AutoModelForVision2Seq.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            )
        if model is None:
            log.warning("florence2: failed to load model")
            self._arch = "florence2_unsupported"
            return 0
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    def _load_cogvlm(self) -> int:
        """Load CogVLM2 model using AutoProcessor + trust_remote_code."""
        log.info("multimodal: loading CogVLM model_id=%s (may download)", self.model_id)
        self.processor = self._safe_call(
            lambda: AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        )
        model = self._safe_call(
            lambda: AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        )
        if model is None:
            log.warning("cogvlm: failed to load model")
            self._arch = "cogvlm_unsupported"
            return 0
        self.model = self._to_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self._count_params(self.model)

    # ---------------------------- Predictors ----------------------------------
    def _predict_blip(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        enc = self.processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_length=self._cap_max_new_tokens(int(options.get("max_length", 32))), do_sample=False, num_beams=1)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return {"answer": answer, "arch": "blip"}

    def _predict_llava(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = self._ensure_image_tokens(question, 1)
            enc = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self._cap_max_new_tokens(int(options.get("max_length", 32))), do_sample=False, num_beams=1)
            if hasattr(self.processor, "batch_decode"):
                text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            elif hasattr(self.processor, "decode"):
                text = self.processor.decode(out[0], skip_special_tokens=True)
            else:
                tok = self._get_tokenizer()
                text = tok.decode(out[0], skip_special_tokens=True)
            return {"answer": text, "arch": "llava"}
        except Exception as e:
            log.error("llava generate failed: %s", e)
            return {}

    def _predict_qwen_vl(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.model = self._safe_call(lambda: _tf.AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True))
            self.model = self._to_device(self.model)
            if hasattr(self.model, "eval"):
                self.model.eval()
            if self.model is None:
                return {}
        chat = getattr(self.model, "chat", None)
        if callable(chat):
            try:
                resp = chat(self.processor, image, question) if self.processor is not None else chat(image, question)
                return {"answer": str(resp), "arch": "qwen_vl"}
            except Exception as e:
                log.error("qwen_vl chat failed: %s", e)
        tok = (self.processor or self.model).tokenizer
        enc = {k: (v.to(self.device) if hasattr(v, "to") and self.device else v) for k, v in tok(question, return_tensors="pt").items()}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self._cap_max_new_tokens(int(options.get("max_length", 32))), do_sample=False, num_beams=1)
        txt = tok.decode(out[0], skip_special_tokens=True)
        return {"answer": txt, "arch": "qwen_vl"}

    def _predict_minicpm(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            return {}
        tok = getattr(self, "tokenizer", None) or getattr(self.model, "tokenizer", None)
        if tok is None:
            return {}
        chat = getattr(self.model, "chat", None)
        msgs = [{"role": "user", "content": question}]
        if callable(chat):
            try:
                out = chat(image=image, msgs=msgs, context=None, tokenizer=tok, sampling=False, max_new_tokens=self._cap_max_new_tokens(int(options.get("max_length", 32))))
                ans = out[0] if isinstance(out, (list, tuple)) and out else out
                return {"answer": str(ans), "arch": "minicpm_vlm_chat"}
            except Exception as e:
                log.error("minicpm chat failed: %s", e)
        try:
            ans = self._minicpm_manual_decode(question, self._cap_max_new_tokens(int(options.get("max_length", 32))) )
            return {"answer": ans, "arch": "minicpm_vlm_manual"}
        except Exception as e:
            log.error("minicpm manual decode failed: %s", e)
            return {}

    def _predict_vlm(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        enc = self._build_vlm_inputs(image, question, options)
        if enc.get("_pipeline_text"):
            return {"answer": enc["_pipeline_text"], "arch": "vlm_pipeline"}
        if enc.get("_skip_generation") or self.model is None:
            log.info("vlm: skip_generation=%s model_none=%s", enc.get("_skip_generation"), self.model is None)
            return {}
        self._strip_processor_only_kwargs(enc)
        try:
            log.info("vlm: starting generate with keys=%s", list(enc.keys()))
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self._cap_max_new_tokens(int(options.get("max_length", 32))), do_sample=False, num_beams=1)
            log.info("vlm: generate completed")
        except Exception as e:
            log.error("vlm: generate failed: %s", e)
            return {}
        answer = self._decode_output(out)
        return {"answer": answer, "arch": "vlm"}

    def _predict_generic_vqa(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        out = self.pipe(image=image, question=question)
        first = out[0] if isinstance(out, list) else out
        ans = first.get("generated_text") or first.get("answer")
        return {"answer": ans, "arch": "generic_vqa"} if ans else {}

    def _predict_yi_vl(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using Yi-VL model. Returns {} if unsupported."""
        if getattr(self, "_arch", None) == "yi_vl_unsupported" or self.model is None:
            log.warning("yi_vl: model unsupported or not loaded")
            return {}
        try:
            question = self._ensure_image_tokens(question, 1)
            enc = self.processor(text=question, images=[image], return_tensors="pt")
            enc = self._move_to_device(enc)
            max_tokens = self._cap_max_new_tokens(int(options.get("max_length", 32)))
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
            answer = self._decode_output(out)
            return {"answer": answer, "arch": "yi_vl"}
        except Exception as e:
            log.error("yi_vl predict failed: %s", e)
            return {}

    def _predict_internvl(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using InternVL2. Prefer chat-style inference."""
        if getattr(self, "_arch", None) == "internvl_unsupported" or self.model is None:
            log.warning("internvl: model unsupported or not loaded")
            return {}
        # Try chat-style interface first
        chat = getattr(self.model, "chat", None)
        if callable(chat):
            try:
                resp = chat(image=image, question=question)
                return {"answer": str(resp), "arch": "internvl_chat"}
            except Exception as e:
                log.info("internvl chat failed: %s", e)
        try:
            question = self._ensure_image_tokens(question, 1)
            enc = self.processor(text=question, images=[image], return_tensors="pt")
            enc = self._move_to_device(enc)
            max_tokens = self._cap_max_new_tokens(int(options.get("max_length", 32)))
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
            answer = self._decode_output(out)
            return {"answer": answer, "arch": "internvl"}
        except Exception as e:
            log.error("internvl predict failed: %s", e)
            return {}

    def _predict_kosmos2(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using Kosmos-2 model."""
        if getattr(self, "_arch", None) == "kosmos2_unsupported" or self.model is None:
            log.warning("kosmos2: model unsupported or not loaded")
            return {}
        try:
            question = self._ensure_image_tokens(question, 1)
            enc = self.processor(text=question, images=[image], return_tensors="pt")
            enc = self._move_to_device(enc)
            self._strip_processor_only_kwargs(enc)
            max_tokens = self._cap_max_new_tokens(int(options.get("max_length", 32)))
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
            answer = self._decode_output(out)
            return {"answer": answer, "arch": "kosmos2"}
        except Exception as e:
            log.error("kosmos2 predict failed: %s", e)
            return {}

    def _predict_florence2(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using Florence-2. Uses task prompt token for VQA."""
        if getattr(self, "_arch", None) == "florence2_unsupported" or self.model is None:
            log.warning("florence2: model unsupported or not loaded")
            return {}
        try:
            # Florence-2 uses task prompt tokens for VQA
            task_prompt = f"<VQA> {question}"
            enc = self.processor(text=task_prompt, images=[image], return_tensors="pt")
            enc = self._move_to_device(enc)
            self._strip_processor_only_kwargs(enc)
            max_tokens = self._cap_max_new_tokens(int(options.get("max_length", 32)))
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
            answer = self._decode_output(out)
            return {"answer": answer, "arch": "florence2"}
        except Exception as e:
            log.error("florence2 predict failed: %s", e)
            return {}

    def _predict_cogvlm(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using CogVLM2. Prefer chat API if available."""
        if getattr(self, "_arch", None) == "cogvlm_unsupported" or self.model is None:
            log.warning("cogvlm: model unsupported or not loaded")
            return {}
        # Try chat-style interface first
        chat = getattr(self.model, "chat", None)
        if callable(chat):
            try:
                resp = chat(image=image, query=question)
                return {"answer": str(resp), "arch": "cogvlm_chat"}
            except Exception as e:
                log.info("cogvlm chat failed: %s", e)
        try:
            question = self._ensure_image_tokens(question, 1)
            enc = self.processor(text=question, images=[image], return_tensors="pt")
            enc = self._move_to_device(enc)
            max_tokens = self._cap_max_new_tokens(int(options.get("max_length", 32)))
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
            answer = self._decode_output(out)
            return {"answer": answer, "arch": "cogvlm"}
        except Exception as e:
            log.error("cogvlm predict failed: %s", e)
            return {}

    # -------------------------- Input building --------------------------------
    def _build_vlm_inputs(self, image, question: str, options: Dict[str, Any]) -> Dict[str, Any]:
        txt = self._try_gemma_pipeline(image, question, options)
        if txt is not None:
            log.info("gemma pipeline produced text; short-circuit")
            return {"_pipeline_text": txt}
        # For Gemma models, avoid hanging generate if pipeline failed to produce text
        if "gemma" in self.model_id.lower():
            log.info("gemma: pipeline returned no text; skipping generation path")
            return {"_skip_generation": True}
        enc = self._encode_with_processor(image, question)
        if enc is not None:
            return enc
        return self._encode_with_tokenizer(question)

    def _try_gemma_pipeline(self, image, question: str, options: Dict[str, Any]) -> Optional[str]:
        if "gemma" not in self.model_id.lower():
            return None
        dev = 0 if self._is_cuda() else None
        try:
            log.info("gemma: building image-text-to-text pipeline (device=%s)", dev)
            pl = pipeline("image-text-to-text", model=self.model_id, trust_remote_code=True, device=dev)
        except Exception as e:
            log.info("gemma pipeline build failed: %s", e)
            return None
        gen_max = self._cap_max_new_tokens(int(options.get("max_length", 16)))
        try:
            log.info("gemma: calling pipeline with capped max_new_tokens=%d", gen_max)
            result_any = pl(images=[image], text=question, generate_kwargs={"max_new_tokens": gen_max, "do_sample": False, "num_beams": 1})  # type: ignore
            text = self._extract_text(result_any)
            log.info("gemma pipeline returned: %s", (text[:80] + "...") if isinstance(text, str) and len(text) > 80 else text)
            if text:
                return text
        except Exception as e:
            log.info("gemma pipeline call failed: %s", e)
            return None
        return None

    def _encode_with_processor(self, image, question: str) -> Optional[Dict[str, Any]]:
        if self.processor is None:
            return None
        try:
            if any(k in self.model_id.lower() for k in ["idefics2", "idefics-2"]):
                question = self._ensure_image_tokens(question, 1)
            enc = self.processor(text=question, images=[image], return_tensors="pt")
            return self._move_to_device(enc)
        except TypeError:
            try:
                enc = self.processor(text=question, return_tensors="pt")
                return self._move_to_device(enc)
            except Exception as e2:
                log.info("processor text-only failed: %s", e2)
                return None
        except Exception as e:
            log.info("processor encode failed: %s", e)
            enc = self._retry_processor_with_image_tokens(image, question)
            return self._move_to_device(enc) if enc else None

    def _retry_processor_with_image_tokens(self, image, question: str) -> Optional[Dict[str, Any]]:
        candidates = [self._get_image_token(), "<image>", "<img>", "[IMG]", "<Image>"]
        tried, retries, max_retries = set(), 0, 4
        for cand in candidates:
            for tpl in (f"{question} {cand}", f"{cand} {question}", f"{cand}"):
                if tpl in tried:
                    continue
                tried.add(tpl)
                try:
                    enc = self.processor(text=tpl, images=[image], return_tensors="pt")
                    return self._move_to_device(enc)
                except Exception:
                    retries += 1
                    if retries >= max_retries:
                        return {"_skip_generation": True}
        return None

    def _encode_with_tokenizer(self, question: str) -> Dict[str, Any]:
        tok = self._get_tokenizer()
        enc = tok(question, return_tensors="pt")
        return self._move_to_device(enc)

    # ---------------------------- MiniCPM utils -------------------------------
    def _minicpm_manual_decode(self, question: str, max_len: int) -> str:
        tok = self._require(getattr(self, "tokenizer", None), "MiniCPM-V manual decode missing tokenizer")
        llm = getattr(self.model, "llm", None) or self._require(self.model, "MiniCPM-V manual decode missing model")
        device = next(llm.parameters()).device if hasattr(llm, "parameters") else (self.device or torch.device("cpu"))
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tok(question, return_tensors="pt").items()}
        if hasattr(llm, "generate"):
            try:
                out = llm.generate(**enc, max_new_tokens=max_len, do_sample=False, num_beams=1)
                text = tok.decode(out[0], skip_special_tokens=True).strip()
                if text:
                    return text
            except Exception as e:
                log.debug("minicpm manual generate failed: %s", e)
        out = llm(**enc)
        logits = getattr(out, "logits", out[0] if isinstance(out, (list, tuple)) else None)
        if logits is None:
            raise RuntimeError("MiniCPM-V manual decode produced no logits")
        token_id = int(logits[0, -1].argmax().item())
        text = tok.decode([token_id], skip_special_tokens=True).strip()
        return text or (tok.decode([token_id]) or "")

    def _patch_minicpm_generate(self):
        if self.model is None:
            return
        llm = getattr(self.model, "llm", None)
        if llm is None:
            return
        try:
            from transformers.generation.utils import GenerationMixin
        except Exception:  # pragma: no cover
            from transformers.generation import GenerationMixin  # type: ignore
        if GenerationMixin not in llm.__class__.__mro__:
            Patched = type(llm.__class__.__name__ + "Patched", (GenerationMixin, llm.__class__), {})
            llm.__class__ = Patched
            log.info("patched %s with GenerationMixin", llm.__class__.__name__)
        if not hasattr(llm, "generation_config") or llm.generation_config is None:
            try:
                from transformers import GenerationConfig
                from transformers import PretrainedConfig
                base = getattr(llm, "config", None)
                llm.generation_config = GenerationConfig.from_model_config(base) if isinstance(base, PretrainedConfig) else GenerationConfig()
            except Exception:  # pragma: no cover
                from transformers.generation import (
                    GenerationConfig,  # type: ignore
                )
                llm.generation_config = GenerationConfig()
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            for kid in ["bos_token_id", "eos_token_id", "pad_token_id"]:
                val = getattr(tok, kid, None)
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

    # ------------------------------ Helpers -----------------------------------
    def _strip_processor_only_kwargs(self, enc: Dict[str, Any]) -> None:
        if "gemma" in self.model_id.lower():
            enc.pop("num_crops", None)

    def _cap_max_new_tokens(self, requested: int) -> int:
        """Cap max_new_tokens on CPU/MPS to prevent slow/hanging generations for big VLMs."""
        if self._is_cuda():
            return max(1, requested)
        # On CPU/MPS, keep very small caps for responsiveness
        safe_cap = 16
        return max(1, min(requested, safe_cap))

    def _get_image_token(self) -> str:
        tok = self._get_tokenizer()
        im = getattr(tok, "image_token", None)
        if isinstance(im, str) and im:
            return im
        stm = getattr(tok, "special_tokens_map", {}) or {}
        for v in stm.values():
            if isinstance(v, str) and "image" in v.lower():
                return v
        addl = getattr(tok, "additional_special_tokens", None)
        if isinstance(addl, (list, tuple)):
            for a in addl:
                if isinstance(a, str) and "image" in a.lower():
                    return a
        return "<image>"

    def _ensure_image_tokens(self, question: str, num_images: int = 1) -> str:
        token = self._get_image_token()
        if question.count(token) == num_images:
            return question
        q = question.replace(token, "").strip()
        return f"{' '.join([token]*num_images)} {q}".strip()

    def _extract_text(self, out: Any) -> Optional[str]:
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            return out.get("text") or out.get("generated_text") or out.get("answer")
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("text") or first.get("answer")
            if isinstance(first, list) and first and isinstance(first[0], dict):
                return first[0].get("generated_text") or first[0].get("text")
        return None

    def _decode_output(self, out) -> str:
        if self.processor is not None:
            if hasattr(self.processor, "batch_decode"):
                return self.processor.batch_decode(out, skip_special_tokens=True)[0]
            if hasattr(self.processor, "decode"):
                return self.processor.decode(out[0], skip_special_tokens=True)
        tok = self._get_tokenizer()
        return tok.decode(out[0], skip_special_tokens=True)

    def _get_tokenizer(self):
        tok = None
        if getattr(self, "tokenizer", None) is not None:
            tok = getattr(self, "tokenizer")
        elif self.processor is not None and hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
        elif self.model is not None and hasattr(self.model, "tokenizer"):
            tok = getattr(self.model, "tokenizer")
        if tok is None:
            raise RuntimeError("no tokenizer/processor available")
        return tok

    def _move_to_device(self, enc: Any) -> Dict[str, Any]:
        # Accept BatchEncoding/mapping-like and normalize to plain dict with tensors on device
        if not isinstance(enc, dict):
            try:
                enc = dict(enc)
            except Exception:
                # last resort: attempt common accessors
                enc = {k: getattr(enc, k) for k in dir(enc) if not k.startswith("_")}
        if not self.device:
            return enc
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}

    def _to_device(self, model):
        if model is None:
            return None
        try:
            return model.to(self.device) if self.device else model
        except Exception:
            return model

    def _unify_model_dtype(self, model, dtype) -> None:
        if model is None:
            return
        for p in model.parameters():
            if p.dtype != dtype:
                p.data = p.data.to(dtype)
        for name, buf in model.named_buffers():
            if buf.dtype != dtype:
                setattr(model, name, buf.to(dtype))

    def _select_dtype(self):
        if self._is_cuda():
            cc = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
            return torch.bfloat16 if cc >= 8 else torch.float16
        if self.device and self.device.type == "mps":
            return torch.float16
        return torch.float32

    def _count_params(self, model) -> int:
        return sum(p.numel() for p in model.parameters()) if (model is not None and hasattr(model, "parameters")) else 0

    def _require(self, val, err: str):
        if not val:
            raise RuntimeError(f"multimodal: {err}")
        return val

    def _is_cuda(self) -> bool:
        return bool(self.device and getattr(self.device, "type", None) == "cuda")


_TASK_MAP = {"image-text-to-text": ImageTextToTextRunner}

def multimodal_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_MAP[task]

__all__ = ["MULTIMODAL_TASKS", "multimodal_runner_for_task", "ImageTextToTextRunner"]
