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
        "facebook/wav2vec2-base-960h",
    ],
)
def test_asr_backend_present_and_runs(client, model_id):
    # Expect soundfile/libsndfile installed
    import soundfile as sf  # type: ignore

    assert sf is not None

    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "audio",
        "inputs": {"audio_base64": _tiny_wav_base64()},
        "task": "automatic-speech-recognition",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    # Even if transcription text is empty (short tone), backend must work without 500s
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert isinstance(data.get("text"), str)
