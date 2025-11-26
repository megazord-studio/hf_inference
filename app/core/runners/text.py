"""Text runners (Phase 0) - generation, classification, embeddings.

KISS: minimal dependencies; rely on transformers / sentence-transformers.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Type

from .base import BaseRunner

log = logging.getLogger("app.runners.text")

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline,
        TextIteratorStreamer,
    )
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    log.warning(f"Transformers or sentence-transformers import failed: {e}")
    torch = None

TEXT_TASKS = {"text-generation", "text-classification", "embedding"}

class TextGenerationRunner(BaseRunner):
    def load(self) -> int:
        log.info("text: loading model_id=%s", self.model_id)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("text: downloading tokenizer/model if not cached")
        tok = AutoTokenizer.from_pretrained(self.model_id)
        mdl = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = tok
        self.model = mdl.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        prompt = inputs.get("text") or ""
        max_new = int(options.get("max_new_tokens", 50))
        temperature = float(options.get("temperature", 1.0))
        top_p = float(options.get("top_p", 1.0))
        stream = bool(options.get("_stream", False))
        if not prompt:
            return {"text": ""}
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": temperature != 0.0 or top_p < 1.0,
            "temperature": temperature,
            "top_p": top_p,
        }
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        if stream and 'TextIteratorStreamer' in globals():  # streaming path
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            import threading
            def _generate():
                with torch.no_grad():
                    self.model.generate(input_ids, streamer=streamer, **gen_kwargs)
            t = threading.Thread(target=_generate)
            t.start()
            # Collect all tokens (blocking) for non-SSE call fallback
            collected = []
            for piece in streamer:
                collected.append(piece)
            text = (prompt + ''.join(collected)).strip()
            return {"text": text, "generation_kwargs": gen_kwargs, "streamed": True, "tokens": collected}
        # Non-streaming path
        with torch.no_grad():
            out_ids = self.model.generate(input_ids, **gen_kwargs)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {"text": text, "generation_kwargs": gen_kwargs, "streamed": False}

class TextClassificationRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        self.pipe = pipeline("text-classification", model=self.model_id, device=0 if self.device and self.device.type == "cuda" else -1)
        self._loaded = True
        # param count approximation not directly exposed; fall back to model parameter sum
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        text = inputs.get("text") or ""
        if not text:
            return {"labels": []}
        res = self.pipe(text)
        # res is List[{'label':..., 'score':...}] for single string
        return {"labels": res}

class EmbeddingRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        self.model = SentenceTransformer(self.model_id)
        self._loaded = True
        # Robust param count across sentence-transformers versions
        base = getattr(self.model, 'auto_model', None) or getattr(self.model, 'model', None)
        if base is None:
            try:
                # Fallback: aggregate over all modules' parameters
                return sum(p.numel() for p in self.model._modules.values() if hasattr(p, 'parameters'))  # type: ignore[attr-defined]
            except Exception:
                return 0
        return sum(p.numel() for p in base.parameters())

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        text = inputs.get("text") or ""
        if not text:
            return {"embedding": []}
        vec = self.model.encode([text])[0]
        return {"embedding": vec.tolist(), "dim": len(vec)}

_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "text-generation": TextGenerationRunner,
    "text-classification": TextClassificationRunner,
    "embedding": EmbeddingRunner,
}


def runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_TO_RUNNER[task]

__all__ = [
    "TextGenerationRunner",
    "TextClassificationRunner",
    "EmbeddingRunner",
    "runner_for_task",
    "TEXT_TASKS",
]
