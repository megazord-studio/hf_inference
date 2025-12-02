"""Text-to-Speech runner supporting SpeechT5 and Coqui TTS."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import wave
from typing import Any
from typing import Dict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from app.core.runners.base import BaseRunner

try:
    import soundfile as sf
except Exception:
    sf = None

# Optional Coqui TTS
_HAS_TTS = False
try:
    from TTS.api import TTS as CoquiTTS

    _HAS_TTS = True
except Exception:
    pass

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


class TextToSpeechRunner(BaseRunner):
    """TTS runner supporting SpeechT5 and Coqui TTS backends."""

    def load(self) -> int:
        use_speecht5 = "speecht5" in self.model_id.lower()

        if use_speecht5:
            return self._load_speecht5()
        return self._load_coqui()

    def _load_speecht5(self) -> int:
        """Load SpeechT5 model."""
        if torch is None:
            raise RuntimeError("torch unavailable")

        from transformers import SpeechT5ForTextToSpeech
        from transformers import SpeechT5HifiGan
        from transformers import SpeechT5Processor

        self.processor = SpeechT5Processor.from_pretrained(self.model_id)
        cfg = _config_without_gc(self.model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_id, config=cfg
        )
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        )

        if self.device:
            try:
                self.model.to(self.device)
                self.vocoder.to(self.device)
            except Exception:
                pass

        model_device = getattr(self.model, "device", torch.device("cpu"))
        self.spk_embed = torch.randn(
            1, 512, dtype=torch.float32, device=model_device
        )
        self.sample_rate = 16000
        self._tts_backend = "speecht5"
        self._loaded = True

        try:
            return sum(p.numel() for p in self.model.parameters())
        except Exception:
            return 0

    def _load_coqui(self) -> int:
        """Load Coqui TTS model."""
        if not _HAS_TTS:
            # Allow Kokoro path to declare not implemented explicitly
            if self.model_id == "hexgrad/Kokoro-82M":
                raise NotImplementedError(
                    "kokoro_tts_not_implemented: Coqui TTS is not available"
                )
            raise RuntimeError(
                "Coqui TTS library not installed (pip install TTS) "
                "or use model_id=microsoft/speecht5_tts"
            )

        mapping = {
            # XTTS v2 official
            "coqui/XTTS-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
            # Kokoro common HF repos mapped to Coqui model name if available
            "hexgrad/Kokoro-82M": "tts_models/en/ljspeech/kokoro",
            "kokoro": "tts_models/en/ljspeech/kokoro",
        }
        # Allow passing a raw Coqui model name directly via model_id
        tts_name = mapping.get(self.model_id, self.model_id)

        # Prefer explicit model selection first; fall back to no-arg last.
        attempts = [
            ("keyword", lambda: CoquiTTS(model_name=tts_name)),
            ("positional", lambda: CoquiTTS(tts_name)),
            ("no-arg", lambda: CoquiTTS()),
        ]

        self.tts = None
        errors = []
        for label, fn in attempts:
            try:
                self.tts = fn()
                break
            except Exception as e:
                errors.append(f"{label}: {e}")

        if self.tts is None:
            if self.model_id == "hexgrad/Kokoro-82M":
                raise NotImplementedError(
                    "kokoro_tts_not_implemented: model load failed in current environment"
                )
            raise RuntimeError(
                f"TTS model load failed for {self.model_id}: {'; '.join(errors)}"
            )

        synth = getattr(self.tts, "synthesizer", None)
        self.sample_rate = getattr(synth, "output_sample_rate", 24000)
        self._tts_backend = "coqui"
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = inputs.get("text") or ""
        if not text:
            return {"audio_base64": ""}

        if self._tts_backend == "speecht5":
            return self._predict_speecht5(text)
        return self._predict_coqui(text, options)

    def _predict_speecht5(self, text: str) -> Dict[str, Any]:
        """Generate speech using SpeechT5."""
        try:
            proc = self.processor(text=text, return_tensors="pt")
            input_ids = proc["input_ids"]
            if hasattr(input_ids, "to"):
                device = getattr(self.model, "device", torch.device("cpu"))
                input_ids = input_ids.to(device)

            with torch.no_grad():
                speech = self.model.generate_speech(
                    input_ids,
                    speaker_embeddings=self.spk_embed,
                    vocoder=self.vocoder,
                )

            wav = speech.detach().cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"SpeechT5 synthesis failed: {e}")

        return self._encode_wav(wav)

    def _predict_coqui(
        self, text: str, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate speech using Coqui TTS."""
        speaker = options.get("speaker") or options.get("voice")
        language = options.get("language") or options.get("lang")
        # Additional XTTS-v2 optional controls when provided
        temperature = options.get("temperature")
        speed = options.get("speed")
        emotion = options.get("emotion")

        if self.tts is None:
            raise RuntimeError("Coqui TTS model not loaded")

        try:
            # Coqui TTS accepts various optional kwargs depending on the model.
            kwargs: Dict[str, Any] = {}
            if speaker is not None:
                kwargs["speaker"] = speaker
            if language is not None:
                kwargs["language"] = language
            if temperature is not None:
                kwargs["temperature"] = temperature
            if speed is not None:
                kwargs["speed"] = speed
            if emotion is not None:
                kwargs["emotion"] = emotion
            wav = self.tts.tts(text, **kwargs)
        except Exception as e:
            if self.model_id == "hexgrad/Kokoro-82M":
                raise NotImplementedError(f"kokoro_tts_not_implemented: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        wav = wav.astype(np.float32)

        if sf is None:
            raise RuntimeError("soundfile library missing for WAV encoding")

        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "audio_base64": f"data:audio/wav;base64,{b64}",
            "sample_rate": self.sample_rate,
            "num_samples": int(wav.shape[0]),
        }

    def _encode_wav(self, wav: np.ndarray) -> Dict[str, Any]:
        """Encode waveform to base64 WAV."""
        try:
            buf = io.BytesIO()
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            wav_clipped = np.clip(wav, -1.0, 1.0)
            pcm16 = (wav_clipped * 32767.0).astype(np.int16)

            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(pcm16.tobytes())

            b64 = base64.b64encode(buf.getvalue()).decode()
            return {
                "audio_base64": f"data:audio/wav;base64,{b64}",
                "sample_rate": self.sample_rate,
                "num_samples": int(pcm16.shape[-1]),
            }
        except Exception as e:
            raise RuntimeError(f"TTS encoding failed: {e}")
