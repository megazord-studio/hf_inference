"""Phase H retrieval & embedding runners.

Implements visual document retrieval using an image-embedding or zero-shot
classification model and an in-memory corpus of document-like labels. Corpus
is constructed at load time and lives entirely in RAM.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Type

import torch
from PIL import Image
from transformers import CLIPImageProcessor
from transformers import CLIPModel

from app.core.utils.media import decode_image_base64

from .base import BaseRunner

RETRIEVAL_TASKS: Set[str] = {"visual-document-retrieval"}


class VisualDocumentRetrievalRunner(BaseRunner):
    """Visual document retrieval over an in-memory corpus."""

    # Attribute annotations for mypy clarity
    processor: Any
    model: Any
    corpus_emb: Optional[torch.Tensor]
    _mode: str
    doc_ids: List[str]
    num_docs: int

    def load(self) -> int:
        model_id = self.model_id
        if not model_id:
            raise RuntimeError("visual_document_retrieval_missing_model_id")

        # Try CLIP path first
        try:
            log = logging.getLogger("app.runners.retrieval")
            log.info(
                "retrieval: loading CLIP model_id=%s (may download)", model_id
            )
            self.processor = CLIPImageProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            self._mode = "clip"
        except Exception:
            # Fallback: treat as zero-shot image classification model
            from transformers import AutoImageProcessor
            from transformers import AutoModelForImageClassification

            try:
                log = logging.getLogger("app.runners.retrieval")
                log.info(
                    "retrieval: loading zero-shot image classification model_id=%s (may download)",
                    model_id,
                )
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_id
                )
                if not getattr(self.model.config, "id2label", None):
                    raise RuntimeError(
                        "visual_document_retrieval_missing_id2label"
                    )
                self._mode = "zshot_cls"
            except Exception as e:
                raise RuntimeError(
                    f"visual_document_retrieval_unsupported_model:{model_id}:{e}"
                ) from e

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
                # Use getattr to avoid mypy accessing attribute on a union type without narrowing
                visual_proj = getattr(self.model, "visual_projection", None)
                if visual_proj is None or not hasattr(
                    visual_proj, "out_features"
                ):
                    raise RuntimeError(
                        "visual_document_retrieval_missing_projection"
                    )
                hidden = int(getattr(visual_proj, "out_features"))
            except Exception as e:
                raise RuntimeError(
                    f"visual_document_retrieval_missing_projection:{e}"
                ) from e

            # Use CPU generator for reproducibility, then move tensor to model.device
            cpu_gen = torch.Generator(device="cpu").manual_seed(0)
            corpus_cpu = torch.randn(
                self.num_docs, hidden, generator=cpu_gen, device="cpu"
            )
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
            self.doc_ids: List[str] = [
                f"doc-{i}" for i in range(self.num_docs)
            ]
        else:
            # Zero-shot classification mode: labels are our "documents".
            id2label = self.model.config.id2label
            self.doc_ids = [id2label[i] for i in sorted(id2label.keys())]
            self.num_docs = len(self.doc_ids)
            self.corpus_emb = None  # not used in classification mode

        self._backend = "hf-visual-retrieval"
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def _encode_image_clip(self, image: Image.Image) -> torch.Tensor:
        enc = self.processor(images=image, return_tensors="pt")
        # enc is BatchFeature (dict-like); cast to dict of tensors for mypy by comprehension
        enc_dict = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in enc.items()
        }
        with torch.no_grad():
            out = self.model.get_image_features(**enc_dict)
        emb = torch.nn.functional.normalize(out, dim=-1)
        return emb[0]

    def _scores_zshot_cls(self, image: Image.Image) -> torch.Tensor:
        from torch.nn.functional import softmax

        enc = self.processor(images=image, return_tensors="pt")
        enc_dict = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in enc.items()
        }
        with torch.no_grad():
            out = self.model(**enc_dict)
        logits = out.logits[0]
        return softmax(logits, dim=-1)

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not isinstance(img_b64, str) or not img_b64.strip():
            raise RuntimeError("visual_document_retrieval_missing_image")

        image = decode_image_base64(img_b64).convert("RGB")

        k = int(options.get("k", 3))
        k = max(1, min(k, self.num_docs))

        if getattr(self, "_mode", None) == "clip":
            if self.corpus_emb is None:
                raise RuntimeError("visual_document_retrieval_corpus_missing")
            q = self._encode_image_clip(image)  # (D,)
            sims = torch.matmul(self.corpus_emb, q)  # (N,)
            values, indices = torch.topk(sims, k)
        else:
            scores = self._scores_zshot_cls(image)  # (num_labels,)
            values, indices = torch.topk(scores, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            doc_id = self.doc_ids[int(idx)]
            results.append({"doc_id": doc_id, "score": float(score)})

        return {"results": results}


_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "visual-document-retrieval": VisualDocumentRetrievalRunner,
}


def retrieval_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_TO_RUNNER[task]


__all__ = [
    "RETRIEVAL_TASKS",
    "retrieval_runner_for_task",
    "VisualDocumentRetrievalRunner",
]
