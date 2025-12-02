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

        log = logging.getLogger("app.runners.retrieval")

        # Try generic VLM/CLIP-like encoder with trust_remote_code
        from transformers import AutoModel, AutoProcessor, AutoImageProcessor
        try:
            log.info(
                "retrieval: loading AutoProcessor/AutoModel (trust_remote_code) model_id=%s",
                model_id,
            )
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(model_id)
            try:
                self.model = AutoModel.from_pretrained(
                    model_id, trust_remote_code=True
                )
            except TypeError:
                self.model = AutoModel.from_pretrained(model_id)

            # Assume VLM path; _encode_image will probe callable interfaces at runtime
            self._mode = "vl_embed"
        except Exception:
            # Try CLIP baseline
            try:
                log.info(
                    "retrieval: loading CLIP model_id=%s (may download)",
                    model_id,
                )
                self.processor = CLIPImageProcessor.from_pretrained(model_id)
                self.model = CLIPModel.from_pretrained(model_id)
                self._mode = "clip"
            except Exception:
                # Fallback: treat as zero-shot image classification model
                from transformers import AutoModelForImageClassification

                try:
                    log.info(
                        "retrieval: loading zero-shot image classification model_id=%s (may download)",
                        model_id,
                    )
                    self.processor = AutoImageProcessor.from_pretrained(
                        model_id
                    )
                    self.model = (
                        AutoModelForImageClassification.from_pretrained(
                            model_id
                        )
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
        # Some remote-code embedding models require an explicit task selection
        # prior to encode calls (e.g., JinaEmbeddingsV4). If exposed, set it.
        if hasattr(self.model, "task"):
            try:
                setattr(self.model, "task", "retrieval")
            except Exception:
                pass

        if self._mode in ("clip", "vl_embed"):
            # Build a tiny in-memory corpus: random unit vectors representing docs.
            self.num_docs = 5
            hidden: Optional[int] = None

            # 1. Direct projection layer (e.g. CLIP visual_projection)
            visual_proj = getattr(self.model, "visual_projection", None)
            if hidden is None and visual_proj is not None and hasattr(visual_proj, "out_features"):
                try:
                    hidden = int(getattr(visual_proj, "out_features"))
                except Exception:
                    hidden = None

            # 2. Common config field (projection_dim)
            if hidden is None:
                cfg = getattr(self.model, "config", None)
                for attr in ("projection_dim", "multi_vector_projector_dim", "hidden_size"):
                    if cfg is not None and hasattr(cfg, attr):
                        try:
                            hidden = int(getattr(cfg, attr))
                            if hidden > 0:
                                break
                        except Exception:
                            hidden = None

            # 3. Probe by encoding a dummy image (robust, last resort)
            if hidden is None:
                try:
                    dummy = Image.new("RGB", (32, 32), (128, 128, 128))
                    feat = self._encode_image(dummy)
                    hidden = int(feat.shape[-1])
                except Exception as e:
                    raise RuntimeError(
                        f"visual_document_retrieval_infer_hidden_failed:{e}"
                    ) from e

            # Hidden must be established by now
            cpu_gen = torch.Generator(device="cpu").manual_seed(0)
            corpus_cpu = torch.randn(self.num_docs, hidden, generator=cpu_gen, device="cpu")
            target_device = getattr(self.model, "device", None)
            if target_device is not None:
                try:
                    corpus_cpu = corpus_cpu.to(target_device)
                except Exception:
                    pass
            corpus_cpu = torch.nn.functional.normalize(corpus_cpu, dim=-1)
            self.corpus_emb = corpus_cpu  # (N, D)
            self.doc_ids = [f"doc-{i}" for i in range(self.num_docs)]
        else:
            # Zero-shot classification mode: labels are our "documents".
            id2label = self.model.config.id2label
            self.doc_ids = [id2label[i] for i in sorted(id2label.keys())]
            self.num_docs = len(self.doc_ids)
            self.corpus_emb = None  # not used in classification mode

        self._backend = "hf-visual-retrieval"
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        # Some remote-code processors (e.g., Qwen2VL-based) require a text prompt
        # when using the full AutoProcessor. Prefer the dedicated image_processor
        # if available to avoid text requirements for pure image embedding. If not
        # present, attempt to load an AutoImageProcessor directly.
        img_proc = getattr(self.processor, "image_processor", None)
        if img_proc is None:
            try:
                from transformers import AutoImageProcessor as _AIP

                img_proc = _AIP.from_pretrained(self.model_id)
            except Exception:
                img_proc = None

        if img_proc is not None:
            enc = img_proc(images=image, return_tensors="pt")
        else:
            enc = self.processor(images=image, return_tensors="pt")
        enc_dict = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in enc.items()
        }

        with torch.no_grad():
            out = None
            # 1. Dedicated embedding method
            if out is None and hasattr(self.model, "embed"):
                # Try pixel-tensor signature first
                try:
                    candidate = self.model.embed(**enc_dict)
                except Exception:
                    # Try raw image signature
                    try:
                        candidate = self.model.embed(image=image)
                    except Exception:
                        candidate = None
                if isinstance(candidate, torch.Tensor):
                    out = candidate
                elif isinstance(candidate, dict):
                    for key in ("image_embeds", "image_features", "embeddings", "pooler_output"):
                        if key in candidate and isinstance(candidate[key], torch.Tensor):
                            out = candidate[key]
                            break
                    if out is None:
                        # Fallback: first tensor value in dict
                        for v in candidate.values():
                            if isinstance(v, torch.Tensor):
                                out = v
                                break
            # 2. Standard image feature helpers
            if out is None and hasattr(self.model, "get_image_features"):
                # Try kwargs form, else positional
                try:
                    out = self.model.get_image_features(**enc_dict)
                except TypeError:
                    # Pass only pixel tensor or raw image
                    pv = enc_dict.get("pixel_values")
                    if pv is not None:
                        out = self.model.get_image_features(pv)
                    else:
                        out = self.model.get_image_features(image)
            elif out is None and hasattr(self.model, "encode_image"):
                try:
                    out = self.model.encode_image(**enc_dict)
                except TypeError:
                    # Prefer raw PIL image list for remote-code models (e.g., Jina)
                    try:
                        out = self.model.encode_image(images=[image])
                    except Exception:
                        pv = enc_dict.get("pixel_values")
                        if pv is not None:
                            out = self.model.encode_image(pv)
                        else:
                            # Some models expect keyword 'image'
                            out = self.model.encode_image(image=image)
            # 3. Vision backbone direct
            elif out is None and hasattr(self.model, "vision_model"):
                out_vm = self.model.vision_model(**enc_dict)
                last = getattr(out_vm, "last_hidden_state", None)
                if last is None:
                    raise RuntimeError("visual_document_retrieval_missing_last_hidden")
                out = last[:, 0] if last.shape[1] > 1 else last.mean(dim=1)
            # 4. Generic forward probing
            if out is None:
                fwd = self.model(**enc_dict)
                if isinstance(fwd, dict):
                    for key in ("image_embeds", "image_features", "pooler_output"):
                        if key in fwd and isinstance(fwd[key], torch.Tensor):
                            out = fwd[key]
                            break
                    if out is None:
                        # Maybe hidden state tensor lives under a key
                        for v in fwd.values():
                            if isinstance(v, torch.Tensor):
                                out = v
                                break
                if out is None:
                    last = getattr(fwd, "last_hidden_state", None)
                    if last is not None:
                        out = last[:, 0] if last.shape[1] > 1 else last.mean(dim=1)
                if out is None:
                    raise RuntimeError("visual_document_retrieval_no_image_encoder_runtime")

        # If model returned a tuple or list, choose the first tensor-like output
        if isinstance(out, (tuple, list)):
            for elem in out:
                if isinstance(elem, torch.Tensor):
                    out = elem
                    break
        # Multi-vector → mean pool across vector axis
        if out.ndim == 3:
            out = out.mean(dim=1)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        # Ensure floating type suitable for cosine
        if out.dtype not in (torch.float32, torch.float64):
            try:
                out = out.float()
            except Exception:
                pass
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

        if getattr(self, "_mode", None) in ("clip", "vl_embed"):
            if self.corpus_emb is None:
                raise RuntimeError("visual_document_retrieval_corpus_missing")
            q = self._encode_image(image)  # (D,)
            # Some remote-code models (e.g., Jina V4) can return higher-dim embeddings
            # than the corpus vectors (e.g., 2048 vs 128 via Matryoshka dims). Align dims.
            corpus_dim = self.corpus_emb.shape[-1]
            if q.shape[-1] != corpus_dim:
                q = q[..., :corpus_dim]
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
