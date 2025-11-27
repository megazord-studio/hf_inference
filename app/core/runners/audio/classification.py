"""Audio Classification runner."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any
from typing import Dict

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification

from app.core.runners.base import BaseRunner

from .utils import decode_base64_audio
from .utils import get_target_sample_rate
from .utils import normalize_audio
from .utils import resample_audio

try:
    import soundfile as sf
except Exception:
    sf = None

log = logging.getLogger("app.runners.audio")


def _config_without_gc(model_id: str) -> AutoConfig:
    """Load config, strip gradient_checkpointing."""
    try:
        cfg_path = hf_hub_download(model_id, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if "gradient_checkpointing" in raw:
            raw.pop("gradient_checkpointing", None)
            tmp_dir = tempfile.mkdtemp(prefix="cfg_gc_strip_")
            with open(os.path.join(tmp_dir, "config.json"), "w") as out:
                json.dump(raw, out)
            cfg = AutoConfig.from_pretrained(tmp_dir, local_files_only=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return cfg
        return AutoConfig.from_pretrained(model_id)
    except Exception:
        return AutoConfig.from_pretrained(model_id)


class AudioClassificationRunner(BaseRunner):
    """Audio classification runner."""

    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")

        self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        cfg = _config_without_gc(self.model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_id, config=cfg
        )

        if self.device:
            self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        audio_b64 = inputs.get("audio_base64")
        if not audio_b64:
            return {"predictions": []}

        try:
            bio = decode_base64_audio(audio_b64)
            if sf is None:
                return {"predictions": []}

            data, sr = sf.read(bio)
            data = normalize_audio(data)

            target_sr = get_target_sample_rate(self.processor)
            data, sr = resample_audio(data, sr, target_sr)

            enc = self.processor(data, sampling_rate=sr, return_tensors="pt")

            with torch.no_grad():
                out = self.model(
                    **{k: v.to(self.model.device) for k, v in enc.items()}
                )
                probs = out.logits.softmax(-1)[0]

            requested_k = int(options.get("top_k", 3))
            top_k = max(1, min(requested_k, probs.shape[-1]))
            values, indices = probs.topk(top_k)

            labels = [self.model.config.id2label[i.item()] for i in indices]
            return {
                "predictions": [
                    {"label": l, "score": float(v.item())}
                    for l, v in zip(labels, values)
                ]
            }
        except Exception as e:
            log.warning("audio-classification predict error: %s", e)
            return {"predictions": []}
