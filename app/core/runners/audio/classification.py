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

# Optional SpeechBrain (v1+)
_HAS_SPEECHBRAIN = False
SBEncoderClassifierCls: Any | None = None
try:
    from speechbrain.inference import EncoderClassifier as _SBEncoderClassifier

    SBEncoderClassifierCls = _SBEncoderClassifier
    _HAS_SPEECHBRAIN = True
except Exception:
    SBEncoderClassifierCls = None

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

        # Heuristic: SpeechBrain models live under the 'speechbrain/' org,
        # and usually don't provide a transformers feature extractor.
        if self.model_id.startswith("speechbrain/"):
            if not _HAS_SPEECHBRAIN:
                raise RuntimeError(
                    "speechbrain library not installed for SpeechBrain model"
                )
            self._backend = "speechbrain"
            if SBEncoderClassifierCls is None:
                raise RuntimeError(
                    "speechbrain import failed for SpeechBrain model"
                )
            self.sb_classifier = SBEncoderClassifierCls.from_hparams(
                source=self.model_id,
                run_opts={
                    "device": str(self.device) if self.device else "cpu"
                },
            )
            # Default target sample rate for SB emotion model
            self._sb_sr = 16000
            self._loaded = True
            return 0

        # Default transformers pathway
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        cfg = _config_without_gc(self.model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_id, config=cfg
        )

        if self.device:
            self.model.to(self.device)
        self.model.eval()
        self._backend = "transformers"
        self._loaded = True
        return sum(p.numel() for p in self.model.parameters())

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        audio_b64 = inputs.get("audio_base64")
        if not audio_b64:
            return {"predictions": []}

        try:
            bio = decode_base64_audio(audio_b64)
            if sf is None:
                return {"predictions": []}

            data, sr = sf.read(bio)
            data = normalize_audio(data)

            backend = getattr(self, "_backend", "transformers")
            if backend == "speechbrain":
                # Resample to SB expected SR
                target_sr = getattr(self, "_sb_sr", 16000)
                data, sr = resample_audio(data, sr, target_sr)
                wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = self.sb_classifier.classify_batch(wav)
                # out.scores: [batch, classes], out.prediction: [batch]
                scores = out.scores.softmax(-1)[0]
                # Map indices to labels across SB versions
                le = getattr(self.sb_classifier.hparams, "label_encoder", None)

                def _idx_to_label(i: int) -> str:
                    try:
                        if le is None:
                            return str(int(i))
                        ind2lab = getattr(le, "ind2lab", None)
                        if ind2lab is not None:
                            if isinstance(ind2lab, (list, tuple)):
                                if 0 <= int(i) < len(ind2lab):
                                    return str(ind2lab[int(i)])
                            else:
                                return str(ind2lab.get(int(i), str(int(i))))
                        decode_ndim = getattr(le, "decode_ndim", None)
                        if callable(decode_ndim):
                            import torch as _t

                            return str(decode_ndim(_t.tensor([int(i)]))[0])
                    except Exception:
                        pass
                    return str(int(i))

                requested_k = int(options.get("top_k", 3))
                top_k = max(1, min(requested_k, scores.shape[-1]))
                values, indices = torch.topk(scores, k=top_k)
                labels = [_idx_to_label(int(i)) for i in indices]
                return {
                    "predictions": [
                        {"label": lbl, "score": float(v)}
                        for lbl, v in zip(labels, values)
                    ]
                }
            else:
                target_sr = get_target_sample_rate(self.processor)
                data, sr = resample_audio(data, sr, target_sr)
                enc = self.processor(
                    data, sampling_rate=sr, return_tensors="pt"
                )
                with torch.no_grad():
                    out = self.model(
                        **{k: v.to(self.model.device) for k, v in enc.items()}
                    )
                    probs = out.logits.softmax(-1)[0]
                requested_k = int(options.get("top_k", 3))
                top_k = max(1, min(requested_k, probs.shape[-1]))
                values, indices = probs.topk(top_k)
                labels = [
                    self.model.config.id2label[i.item()] for i in indices
                ]
                return {
                    "predictions": [
                        {"label": lbl, "score": float(v.item())}
                        for lbl, v in zip(labels, values)
                    ]
                }
        except Exception as e:
            log.warning("audio-classification predict error: %s", e)
            return {"predictions": []}
