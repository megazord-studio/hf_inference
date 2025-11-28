"""Text runners (Phase 0) - generation, classification, embeddings.

KISS: minimal dependencies; rely on transformers / sentence-transformers.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Any as _AnyType
from typing import Dict
from typing import Optional
from typing import Type

from .base import BaseRunner

log = logging.getLogger("app.runners.text")

torch: _AnyType | None = None

try:
    import torch as _torch

    torch = _torch
    from sentence_transformers import SentenceTransformer
    from transformers import TextIteratorStreamer
    from transformers import pipeline
except Exception as e:  # pragma: no cover
    log.warning(f"Transformers or sentence-transformers import failed: {e}")
    torch = None

TEXT_TASKS = {
    "text-generation",
    "text-classification",
    "embedding",
    "summarization",
}


class TextGenerationRunner(BaseRunner):
    def load(self) -> int:
        log.info("text: loading model_id=%s", self.model_id)
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        log.info("text: downloading tokenizer/model if not cached")
        tok = AutoTokenizer.from_pretrained(self.model_id)
        mdl = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = tok
        self.model = mdl.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return sum(p.numel() for p in self.model.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch unavailable")
        prompt = inputs.get("text") or ""
        if not prompt:
            return {"text": ""}
        max_new = int(options.get("max_new_tokens", 50))
        temperature = float(options.get("temperature", 1.0))
        top_p = float(options.get("top_p", 1.0))
        stream = bool(options.get("_stream", False))
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": temperature != 0.0 or top_p < 1.0,
            "temperature": temperature,
            "top_p": top_p,
        }
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        if stream and "TextIteratorStreamer" in globals():  # streaming path
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            import threading

            def _generate() -> None:
                with torch.no_grad():
                    self.model.generate(
                        input_ids, streamer=streamer, **gen_kwargs
                    )

            t = threading.Thread(target=_generate)
            t.start()
            # Collect all tokens (blocking) for non-SSE call fallback
            collected: list[str] = []
            for piece in streamer:
                collected.append(piece)
            text = (prompt + "".join(collected)).strip()
            return {
                "text": text,
                "generation_kwargs": gen_kwargs,
                "streamed": True,
                "tokens": collected,
            }
        # Non-streaming path
        with torch.no_grad():
            out_ids = self.model.generate(input_ids, **gen_kwargs)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {
            "text": text,
            "generation_kwargs": gen_kwargs,
            "streamed": False,
        }


class TextClassificationRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        self.pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=0 if self.device and self.device.type == "cuda" else -1,
        )
        self._loaded = True
        # param count approximation not directly exposed; fall back to model parameter sum
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        base: Optional[Any] = getattr(
            self.model, "auto_model", None
        ) or getattr(self.model, "model", None)
        if base is None:
            try:
                total = 0
                for module in self.model._modules.values():
                    if hasattr(module, "parameters"):
                        for p in module.parameters():
                            total += p.numel()
                return total
            except Exception:
                return 0
        return sum(p.numel() for p in base.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = inputs.get("text") or ""
        if not text:
            return {"embedding": []}
        vec = self.model.encode([text])[0]
        return {"embedding": vec.tolist(), "dim": len(vec)}


class SummarizationRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        from transformers import AutoModelForSeq2SeqLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(
            self.device
        )
        if hasattr(self.model, "eval"):
            self.model.eval()
        return sum(p.numel() for p in self.model.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch unavailable")
        text = inputs.get("text") or ""
        if not text:
            return {"summary_text": ""}
        max_new = int(options.get("max_new_tokens", 60))
        min_new = int(options.get("min_new_tokens", 10))
        do_sample = bool(options.get("do_sample", False))
        temperature = float(options.get("temperature", 1.0))
        inputs_tok = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_new,
            "min_new_tokens": min_new,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        with torch.no_grad():
            output_ids = self.model.generate(**inputs_tok, **gen_kwargs)
        summary = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return {
            "summary_text": summary.strip(),
            "generation_kwargs": gen_kwargs,
        }


_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "text-generation": TextGenerationRunner,
    "text-classification": TextClassificationRunner,
    "embedding": EmbeddingRunner,
    "summarization": SummarizationRunner,
}


def runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_TO_RUNNER[task]


__all__ = [
    "TextGenerationRunner",
    "TextClassificationRunner",
    "EmbeddingRunner",
    "SummarizationRunner",
    "runner_for_task",
    "TEXT_TASKS",
]
