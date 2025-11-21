"""ONNX text generation runner fallback.

Loads an ONNX exported GPT2-like model using onnxruntime.
Simplified: assumes presence of model.onnx and tokenizer.json or vocab files.
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any, Optional, List
import numpy as np  # type: ignore

log = logging.getLogger("app.runners.text_onnx")

try:
    import onnxruntime as ort  # type: ignore
except Exception as e:  # pragma: no cover
    ort = None
    log.warning(f"onnxruntime not available: {e}")

try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
    AutoTokenizer = None
    log.warning(f"transformers import failed: {e}")

try:
    from huggingface_hub import HfApi, hf_hub_download
except Exception as e:  # pragma: no cover
    HfApi = None
    hf_hub_download = None
    log.warning(f"huggingface_hub import failed for ONNX runner: {e}")

from .base import BaseRunner

class OnnxTextGenerationRunner(BaseRunner):
    backend = "onnx"

    def load(self) -> int:
        if ort is None:
            raise RuntimeError("onnxruntime unavailable; install onnxruntime or onnxruntime-gpu")
        if AutoTokenizer is None:
            raise RuntimeError("transformers unavailable for tokenizer")
        # Candidate filenames we prefer
        preferred: List[str] = ["model.onnx", "gpt2.onnx", "decoder_model.onnx"]
        found_path: Optional[str] = None
        # 1) Local search
        for fname in preferred:
            path = self._maybe_local_file(fname)
            if path:
                found_path = path
                break
        # 2) Remote search if not found locally and hub available
        if not found_path and HfApi and hf_hub_download:
            try:
                api = HfApi(token=os.getenv("HF_TOKEN"))
                files = api.list_repo_files(self.model_id)
                # Filter .onnx files
                onnx_files = [f for f in files if f.lower().endswith('.onnx')]
                # Choose best match (preferred order else first)
                match = next((p for p in preferred if p in onnx_files), None) or (onnx_files[0] if onnx_files else None)
                if match:
                    found_path = hf_hub_download(self.model_id, filename=match, token=os.getenv("HF_TOKEN"))
            except Exception as e:  # pragma: no cover
                log.warning(f"Could not list/download ONNX files for {self.model_id}: {e}")
        if not found_path:
            raise RuntimeError(f"No ONNX file found for {self.model_id}; looked for {preferred} or any *.onnx in repo")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._cuda_available() else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(found_path, providers=providers)  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"Failed creating ONNX session for {found_path}: {e}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self._loaded = True
        return 0  # param count unknown for ONNX fallback

    def _maybe_local_file(self, fname: str) -> Optional[str]:
        # First try Hugging Face hub snapshot directory if cached by transformers
        cache_dir = os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        if os.path.isdir(cache_dir):
            for root, _dirs, files in os.walk(cache_dir):
                if fname in files and self.model_id.replace("/", "-") in root:
                    return os.path.join(root, fname)
        # Fallback: check working directory
        local = os.path.join(os.getcwd(), fname)
        return local if os.path.isfile(local) else None

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        text = inputs.get("text") or ""
        if not text:
            return {"text": ""}
        max_new = int(options.get("max_new_tokens", 50))
        temperature = float(options.get("temperature", 1.0))
        top_p = float(options.get("top_p", 1.0))
        top_k = int(options.get("top_k", 0))
        # Initial token ids (list[int])
        input_ids = self.tokenizer.encode(text)
        # Session input names for dynamic feed building
        input_names = {i.name for i in self.session.get_inputs()}  # type: ignore[attr-defined]

        def build_feed(ids: List[int]) -> Dict[str, Any]:
            arr = np.array(ids, dtype=np.int64)[None, :]  # shape (1, seq)
            feed: Dict[str, Any] = {}
            if 'input_ids' in input_names:
                feed['input_ids'] = arr
            if 'attention_mask' in input_names:
                feed['attention_mask'] = np.ones_like(arr, dtype=np.int64)
            if 'position_ids' in input_names:
                feed['position_ids'] = np.arange(arr.shape[1], dtype=np.int64)[None, :]
            # Ignore past_key_values for simplicity (not in exported minimal models)
            return feed

        def sample_next(logits_row: np.ndarray) -> int:
            # logits_row shape (vocab_size,)
            if temperature != 1.0 and temperature > 0:
                logits_row = logits_row / temperature
            # Top-k filtering
            if top_k > 0 and top_k < logits_row.shape[0]:
                k_indices = np.argpartition(-logits_row, top_k)[:top_k]
                mask = np.full_like(logits_row, -1e10)
                mask[k_indices] = logits_row[k_indices]
                logits_row = mask
            # Softmax
            probs = np.exp(logits_row - np.max(logits_row))
            probs /= np.sum(probs)
            # Top-p filtering
            if top_p < 1.0:
                sorted_idx = np.argsort(-probs)
                cumulative = np.cumsum(probs[sorted_idx])
                keep_mask = cumulative <= top_p
                if not np.any(keep_mask):
                    keep_mask[0] = True  # always keep at least one
                keep_indices = sorted_idx[keep_mask]
                filtered = np.zeros_like(probs)
                filtered[keep_indices] = probs[keep_indices]
                probs = filtered / np.sum(filtered)
            # Greedy if temperature=0
            if temperature == 0.0:
                return int(np.argmax(probs))
            return int(np.random.choice(len(probs), p=probs))

        # Generation loop
        outputs = None
        for _ in range(max_new):
            feed = build_feed(input_ids)
            try:
                outputs = self.session.run(None, feed)  # type: ignore[attr-defined]
            except Exception as e:
                log.warning(f"ONNX inference failed: {e}")
                break
            if not outputs:
                break
            logits = outputs[0]
            # Expect shape (batch, seq, vocab); take last position
            if logits.ndim == 3:
                next_logits = logits[0, -1]
            elif logits.ndim == 2:  # (batch, vocab)
                next_logits = logits[0]
            else:
                log.warning("Unexpected logits shape; stopping generation")
                break
            next_id = sample_next(next_logits)
            input_ids.append(next_id)
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return {
            "text": decoded,
            "backend": "onnx",
            "approximate": True,
            "tokens_generated": len(input_ids),
            "initial_length": len(self.tokenizer.encode(text)),
            "parameters": {
                "max_new_tokens": max_new,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        }

    def unload(self) -> None:
        self.session = None  # type: ignore

    def _cuda_available(self) -> bool:
        try:
            return ort and 'CUDAExecutionProvider' in ort.get_available_providers()  # type: ignore[attr-defined]
        except Exception:
            return False

__all__ = ["OnnxTextGenerationRunner"]
