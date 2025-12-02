import base64
import io
import wave

import pytest


def _tiny_wav_base64(seconds: float = 0.1, sr: int = 16000) -> str:
    import numpy as np
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    # encode as PCM16 WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        pcm16 = (wav * 32767.0).astype("int16")
        w.writeframes(pcm16.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


@pytest.mark.parametrize(
    "model_id",
    [
        # Use a common ASR model id; test focuses on preflight, so load will error before download when soundfile missing
        "facebook/wav2vec2-base",
    ],
)
def test_asr_preflight_missing_soundfile_returns_clear_error(client, monkeypatch, model_id):
    # Force soundfile backend to be unavailable
    import app.core.runners.audio.asr as asr_mod
    monkeypatch.setattr(asr_mod, "sf", None, raising=False)

    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "audio",
        "inputs": {"audio_base64": _tiny_wav_base64()},
        "task": "automatic-speech-recognition",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 500, resp.text
    body = resp.json()
    err = body.get("error", {})
    assert err.get("code") == "inference_failed"
    # Message should include a clear backend hint
    assert "soundfile" in (err.get("message") or "") or "asr_backend_missing" in (err.get("message") or "")