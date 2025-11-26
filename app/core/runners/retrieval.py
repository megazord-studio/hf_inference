"""Phase H retrieval & embedding runners.

Implements visual document retrieval using an image-embedding or zero-shot
classification model and an in-memory corpus of document-like labels. Corpus
is constructed at load time and lives entirely in RAM.
"""
from __future__ import annotations

from typing import Dict, Any, Type, Set, List
import logging

import torch
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image

from .base import BaseRunner
from app.core.utils.media import decode_image_base64


RETRIEVAL_TASKS: Set[str] = {"visual-document-retrieval"}


class VisualDocumentRetrievalRunner(BaseRunner):
    """Visual document retrieval over an in-memory corpus.

    Two modes are supported based on the HF model_id:
    - If the model is CLIP-compatible (CLIPModel + CLIPImageProcessor), we use
      get_image_features + a random in-memory corpus to rank docs.
    - If the model is a zero-shot image classification model, we treat its
      class labels as the in-memory "documents" and use its predicted scores
      as retrieval scores.

    If neither mode applies, load() fails with a clear RuntimeError.
    """

    def load(self) -> int:  # type: ignore[override]
        model_id = self.model_id
        if not model_id:
            raise RuntimeError("visual_document_retrieval_missing_model_id")

        # Try CLIP path first
        try:
            log = logging.getLogger("app.runners.retrieval")
            log.info("retrieval: loading CLIP model_id=%s (may download)", model_id)
            self.processor = CLIPImageProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            self._mode = "clip"
        except Exception:
            # Fallback: treat as zero-shot image classification model
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            try:
                log = logging.getLogger("app.runners.retrieval")
                log.info("retrieval: loading zero-shot image classification model_id=%s (may download)", model_id)
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForImageClassification.from_pretrained(model_id)
                if not getattr(self.model.config, "id2label", None):
                    raise RuntimeError("visual_document_retrieval_missing_id2label")
                self._mode = "zshot_cls"
            except Exception as e:
                raise RuntimeError(f"visual_document_retrieval_unsupported_model:{model_id}:{e}") from e

        if self.device:
            try:
                self.model.to(self.device)
            except Exception:
                pass
        self.model.eval()

        if self._mode == "clip":
            # Build a tiny in-memory corpus: random unit vectors representing docs.
            self.num_docs = 5
            try:
                hidden = int(self.model.visual_projection.out_features)
            except Exception as e:
                raise RuntimeError(f"visual_document_retrieval_missing_projection:{e}") from e

            # Use CPU generator for reproducibility, then move tensor to model.device
            cpu_gen = torch.Generator(device="cpu").manual_seed(0)
            corpus_cpu = torch.randn(self.num_docs, hidden, generator=cpu_gen, device="cpu")
            target_device = getattr(self.model, "device", None)
            if target_device is not None:
                try:
                    corpus = corpus_cpu.to(target_device)
                except Exception:
                    corpus = corpus_cpu
            else:
                corpus = corpus_cpu
            corpus = torch.nn.functional.normalize(corpus, dim=-1)
            self.corpus_emb = corpus  # (N, D)
            self.doc_ids: List[str] = [f"doc-{i}" for i in range(self.num_docs)]
        else:
            # Zero-shot classification mode: labels are our "documents".
            id2label = self.model.config.id2label
            self.doc_ids = [id2label[i] for i in sorted(id2label.keys())]
            self.num_docs = len(self.doc_ids)

        self._backend = "hf-visual-retrieval"
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def _encode_image_clip(self, image: Image.Image) -> torch.Tensor:
        enc = self.processor(images=image, return_tensors="pt")
        enc = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.get_image_features(**enc)
        emb = torch.nn.functional.normalize(out, dim=-1)
        return emb[0]

    def _scores_zshot_cls(self, image: Image.Image) -> torch.Tensor:
        from torch.nn.functional import softmax

        enc = self.processor(images=image, return_tensors="pt")
        enc = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        logits = out.logits[0]
        return softmax(logits, dim=-1)

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        img_b64 = inputs.get("image_base64")
        if not isinstance(img_b64, str) or not img_b64.strip():
            raise RuntimeError("visual_document_retrieval_missing_image")

        image = decode_image_base64(img_b64).convert("RGB")

        k = int(options.get("k", 3))
        k = max(1, min(k, self.num_docs))

        if getattr(self, "_mode", None) == "clip":
            q = self._encode_image_clip(image)  # (D,)
            sims = torch.matmul(self.corpus_emb, q)  # (N,)
            values, indices = torch.topk(sims, k)
        else:
            scores = self._scores_zshot_cls(image)  # (num_labels,)
            values, indices = torch.topk(scores, k)

        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            doc_id = self.doc_ids[int(idx)]
            results.append({"doc_id": doc_id, "score": float(score)})

        return {"results": results}


_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "visual-document-retrieval": VisualDocumentRetrievalRunner,
}


def retrieval_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_TO_RUNNER[task]


__all__ = ["RETRIEVAL_TASKS", "retrieval_runner_for_task", "VisualDocumentRetrievalRunner"]
