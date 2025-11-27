"""Automatic Speech Recognition runner."""
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
from transformers import AutoProcessor
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor

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
    """Load config, strip gradient_checkpointing, return sanitized AutoConfig."""
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


class AutomaticSpeechRecognitionRunner(BaseRunner):
    """ASR runner using Wav2Vec2 or similar models."""

    def load(self) -> int:
        if torch is None:
            raise RuntimeError("torch unavailable")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        cfg = _config_without_gc(self.model_id)
        try:
            from transformers import AutoModelForCausalLM

            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, config=cfg)
        except Exception:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        if self.device:
            self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        audio_b64 = inputs.get("audio_base64")
        if not audio_b64:
            return {"text": ""}

        if sf is None:
            raise RuntimeError("soundfile not installed for ASR decoding")

        try:
            bio = decode_base64_audio(audio_b64)
            audio, sr = sf.read(bio)
            audio = normalize_audio(audio)

            target_sr = get_target_sample_rate(self.processor)
            audio, sr = resample_audio(audio, sr, target_sr)

            inputs_proc = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            inputs_proc = {
                k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs_proc.items()
            }

            with torch.no_grad():
                logits = self.model(**inputs_proc).logits

            if logits.shape[-1] == self.model.config.vocab_size:
                pred_ids = torch.argmax(logits, dim=-1)
                text = self.processor.batch_decode(pred_ids)[0].strip()
            else:
                text = ""

            return {"text": text}
        except Exception as e:
            log.warning("automatic-speech-recognition predict error: %s", e)
            return {"text": ""}
