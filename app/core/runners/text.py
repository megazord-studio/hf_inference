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
    "fill-mask",
    "question-answering",
    "sentence-similarity",
    "token-classification",
    "table-question-answering",
    "text-ranking",
    "translation",
    "zero-shot-classification",
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
        # Keep backward-compatible "labels" shape expected by tests
        labels = [
            {"label": item.get("label"), "score": float(item.get("score", 0.0))}
            for item in (res if isinstance(res, list) else [res])
        ]
        return {"labels": labels}


class FillMaskRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        self.pipe = pipeline(
            "fill-mask",
            model=self.model_id,
            device=0 if self.device and self.device.type == "cuda" else -1,
        )
        self._loaded = True
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = inputs.get("text") or ""
        if not text:
            return {"predictions": []}
        top_k = int(options.get("top_k", 5))
        res = self.pipe(text, top_k=top_k)
        preds = [
            {"label": item.get("token_str"), "score": float(item.get("score", 0.0))}
            for item in (res if isinstance(res, list) else [res])
        ]
        return {"predictions": preds}


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


class QuestionAnsweringRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        self.pipe = pipeline(
            "question-answering",
            model=self.model_id,
            device=0 if self.device and self.device.type == "cuda" else -1,
        )
        self._loaded = True
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        question = inputs.get("question") or ""
        context = inputs.get("context") or ""
        if not question or not context:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}
        topk = int(options.get("topk", 1))
        res = self.pipe({"question": question, "context": context}, topk=topk)
        # res can be dict or list of dicts depending on topk
        first = res[0] if isinstance(res, list) else res
        return {
            "answer": first.get("answer", ""),
            "score": float(first.get("score", 0.0)),
            "start": int(first.get("start", 0)),
            "end": int(first.get("end", 0)),
        }


class SentenceSimilarityRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_id)
        self._loaded = True
        base = getattr(self.model, "auto_model", None) or getattr(
            self.model, "model", None
        )
        if base is None:
            return 0
        return sum(p.numel() for p in base.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        a = inputs.get("text")
        b = inputs.get("text_pair")
        if not (a and b):
            # Allow alternative: inputs['sentences'] = [a,b]
            sents = inputs.get("sentences")
            if isinstance(sents, list) and len(sents) >= 2:
                a, b = sents[0], sents[1]
        if not (a and b):
            return {"score": 0.0}
        import numpy as np

        emb = self.model.encode([a, b])
        v1, v2 = emb[0], emb[1]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
        score = float(np.dot(v1, v2) / denom)
        return {"score": score}


class TokenClassificationRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        self.pipe = pipeline(
            "token-classification",
            model=self.model_id,
            aggregation_strategy="simple",
            device=0 if self.device and self.device.type == "cuda" else -1,
        )
        self._loaded = True
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = inputs.get("text") or ""
        if not text:
            return {"entities": []}
        agg = options.get("aggregation_strategy")
        if agg:
            self.pipe.aggregate_strategy = agg  # best effort if supported
        res = self.pipe(text)
        items = res if isinstance(res, list) else [res]
        entities = []
        for it in items:
            entities.append(
                {
                    "entity": it.get("entity_group") or it.get("entity"),
                    "score": float(it.get("score", 0.0)),
                    "word": it.get("word"),
                    "start": int(it.get("start", 0)),
                    "end": int(it.get("end", 0)),
                }
            )
        return {"entities": entities}
class TableQuestionAnsweringRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        from transformers import pipeline

        # TAPAS/TAPEX models use the "table-question-answering" pipeline
        self.pipe = pipeline(
            "table-question-answering",
            model=self.model_id,
            device=0 if self.device and self.device.type == "cuda" else -1,
        )
        self._loaded = True
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        question = inputs.get("question") or ""
        table_in = inputs.get("table")
        if not question or table_in is None:
            return {"answer": "", "cells": [], "coordinates": []}
        import pandas as pd

        df = None
        # Accept multiple table formats:
        # 1. list[list]; first row headers, remaining rows values
        # 2. list[dict]; each dict is a row
        # 3. dict[str, list]; column -> list of values
        if isinstance(table_in, list):
            if table_in and all(isinstance(r, dict) for r in table_in):
                df = pd.DataFrame(table_in)
            elif table_in and all(isinstance(r, list) for r in table_in):
                if len(table_in) >= 2:
                    headers = table_in[0]
                    rows = table_in[1:]
                    df = pd.DataFrame(rows, columns=headers)
        elif isinstance(table_in, dict):
            # column -> list
            if all(isinstance(v, list) for v in table_in.values()):
                df = pd.DataFrame(table_in)
        if df is None:
            return {"answer": "", "cells": [], "coordinates": [], "error": "unsupported_table_format"}
        topk = int(options.get("topk", 1))
        try:
            res = self.pipe(table=df, query=question, topk=topk)
        except Exception as e:
            return {"answer": "", "cells": [], "coordinates": [], "error": repr(e)[:200]}
        # HF pipeline returns dict with answer / cells / coordinates (list of cell indices)
        return {
            "answer": res.get("answer", ""),
            "cells": res.get("cells", []),
            "coordinates": res.get("coordinates", []),
        }
class TextRankingRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(self.model_id)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True
        base = getattr(self.model, "model", None)
        if base is None:
            return 0
        return sum(p.numel() for p in base.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        query = inputs.get("query")
        candidates = inputs.get("candidates")
        pairs = inputs.get("pairs")
        items: list[list[str]] = []
        if isinstance(pairs, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in pairs
        ):
            items = [[str(p[0]), str(p[1])] for p in pairs]
        elif isinstance(query, str) and isinstance(candidates, list):
            items = [[query, str(c)] for c in candidates]
        if not items:
            return {"scores": [], "indices": []}
        # CrossEncoder.predict returns scores aligned to the input pairs
        try:
            scores = self.model.predict(items)
        except Exception as e:
            return {"scores": [], "indices": [], "error": repr(e)[:200]}
        scores_list = [float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else list(scores))]
        # Compute ranking indices (descending)
        indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        return {"scores": scores_list, "indices": indices}
class TranslationRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(
            self.device
        )
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch unavailable")
        text = inputs.get("text") or ""
        if not text:
            return {"text": ""}
        src = options.get("src_lang") or inputs.get("src_lang")
        tgt = options.get("tgt_lang") or inputs.get("tgt_lang")
        prefix = ""
        # For T5-style models, language can be indicated via prefix
        if isinstance(src, str) and isinstance(tgt, str):
            prefix = f"translate {src} to {tgt}: "
        inputs_tok = self.tokenizer(
            prefix + text, return_tensors="pt", truncation=True
        ).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": int(options.get("max_new_tokens", 80)),
            "do_sample": bool(options.get("do_sample", False)),
            "temperature": float(options.get("temperature", 1.0)),
        }
        with torch.no_grad():
            out_ids = self.model.generate(**inputs_tok, **gen_kwargs)
        out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {"text": out_text.strip(), "generation_kwargs": gen_kwargs}
class ZeroShotClassificationRunner(BaseRunner):
    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")
        from transformers import pipeline

        self.pipe = pipeline(
            "zero-shot-classification",
            model=self.model_id,
            device=0 if self.device and self.device.type == "cuda" else -1,
        )
        self._loaded = True
        m = self.pipe.model
        return sum(p.numel() for p in m.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = inputs.get("text") or ""
        labels = inputs.get("labels") or []
        if not text or not labels:
            return {"predictions": []}
        multi_label = bool(options.get("multi_label", False))
        hypothesis_template = options.get(
            "hypothesis_template", "This text is about {}."
        )
        res = self.pipe(
            text,
            candidate_labels=labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template,
        )
        # Normalize to TextClassificationOutput: predictions[{label, score}]
        preds = [
            {"label": lbl, "score": float(scr)}
            for lbl, scr in zip(res.get("labels", []), res.get("scores", []))
        ]
        return {"predictions": preds}
_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "text-generation": TextGenerationRunner,
    "text-classification": TextClassificationRunner,
    "embedding": EmbeddingRunner,
    "summarization": SummarizationRunner,
    "fill-mask": FillMaskRunner,
    "question-answering": QuestionAnsweringRunner,
    "sentence-similarity": SentenceSimilarityRunner,
    "token-classification": TokenClassificationRunner,
    "table-question-answering": TableQuestionAnsweringRunner,
    "text-ranking": TextRankingRunner,
    "translation": TranslationRunner,
    "zero-shot-classification": ZeroShotClassificationRunner,
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
