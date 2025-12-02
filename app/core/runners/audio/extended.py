"""Extended audio runners - audio-to-audio, text-to-audio, VAD, etc."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any
from typing import Dict

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor

from app.core.runners.base import BaseRunner

from .utils import decode_base64_audio

try:
    import soundfile as sf
except Exception:
    sf = None

log = logging.getLogger("app.runners.audio")


class AudioToAudioRunner(BaseRunner):
    """Simple denoising using spectral gating on input audio."""

    def load(self) -> int:
        if sf is None or np is None:
            raise RuntimeError("audio stack unavailable")
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        b64 = inputs.get("audio_base64")
        if not b64:
            raise RuntimeError("audio_to_audio_missing_input")

        buf = decode_base64_audio(b64)
        data, sr = sf.read(buf)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Simple noise gate: zero-out low-energy segments
        energy = np.abs(data)
        thresh = float(options.get("energy_threshold", 0.02))
        denoised = np.where(energy < thresh, 0.0, data).astype(np.float32)

        out_buf = io.BytesIO()
        sf.write(out_buf, denoised, sr, format="WAV")
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(
            out_buf.getvalue()
        ).decode("ascii")

        legacy = bool(options.get("_legacy_tuple", False))
        out = {"audio_base64": audio_b64, "sample_rate": sr}
        if legacy:
            return out, {"backend": "denoise-gate"}
        return out


class TextToAudioRunner(BaseRunner):
    """Text-to-audio using simple waveform synthesis (placeholder)."""

    def load(self) -> int:
        if sf is None or np is None:
            raise RuntimeError("audio stack unavailable")
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = inputs.get("text") or ""
        if not text:
            raise RuntimeError("text_to_audio_missing_text")

        # Placeholder: synthesize a simple tone sequence encoding text length
        sr = 16000
        duration = max(0.3, min(len(text) * 0.05, 3.0))
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = float(options.get("base_freq", 440.0))
        sig = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)

        out_buf = io.BytesIO()
        sf.write(out_buf, sig, sr, format="WAV")
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(
            out_buf.getvalue()
        ).decode("ascii")

        legacy = bool(options.get("_legacy_tuple", False))
        out = {"audio_base64": audio_b64, "sample_rate": sr}
        if legacy:
            return out, {"backend": "tone-synth"}
        return out


class AudioTextToTextRunner(BaseRunner):
    """Audio-text-to-text using Wav2Vec2 ASR for transcription."""

    def load(self) -> int:
        if torch is None or sf is None or np is None:
            raise RuntimeError("asr stack unavailable")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            if self.device:
                self.model.to(self.device)
            self.model.eval()
            self.backend = "wav2vec2"
            self._loaded = True
            return sum(p.numel() for p in self.model.parameters())
        except Exception:
            self.processor = None
            self.model = None
            self.backend = "dummy-asr"
            self._loaded = True
            return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        b64 = inputs.get("audio_base64")
        if not b64:
            raise RuntimeError("audio_text_to_text_missing_input")

        buf = decode_base64_audio(b64)
        data, sr = sf.read(buf)
        if data.ndim > 1:
            data = data.mean(axis=1)

        legacy = bool(options.get("_legacy_tuple", False))

        if getattr(self, "backend", None) == "wav2vec2" and self.processor is not None and self.model is not None:
            with torch.no_grad():
                enc = self.processor(data, sampling_rate=sr, return_tensors="pt")
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                ids = torch.argmax(logits, dim=-1)
                text = self.processor.batch_decode(ids)[0].strip()
        else:
            text = options.get("_dummy_text", "")

        out = {"text": text}
        if legacy:
            return out, {"backend": getattr(self, "backend", "unknown")}
        return out


class VoiceActivityDetectionRunner(BaseRunner):
    """Energy-based voice activity detection returning start/end segments."""

    def load(self) -> int:
        if sf is None or np is None:
            raise RuntimeError("audio stack unavailable")
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        b64 = inputs.get("audio_base64")
        if not b64:
            raise RuntimeError("vad_missing_input")

        buf = decode_base64_audio(b64)
        data, sr = sf.read(buf)
        if data.ndim > 1:
            data = data.mean(axis=1)

        frame_ms = float(options.get("frame_ms", 30.0))
        hop_s = frame_ms / 1000.0
        frame_len = int(sr * hop_s)
        energy = np.abs(data)
        thresh = float(options.get("energy_threshold", 0.02))

        segments = []
        in_speech = False
        start = 0.0

        for i in range(0, len(data), frame_len):
            frame = energy[i : i + frame_len]
            t = i / sr
            active = frame.mean() > thresh
            if active and not in_speech:
                in_speech = True
                start = t
            elif not active and in_speech:
                in_speech = False
                segments.append({"start": start, "end": t})

        if in_speech:
            segments.append({"start": start, "end": len(data) / sr})

        legacy = bool(options.get("_legacy_tuple", False))
        out = {"segments": segments, "sample_rate": sr}
        if legacy:
            return out, {"backend": "energy-vad"}
        return out
