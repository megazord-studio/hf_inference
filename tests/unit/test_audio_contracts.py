import base64
import io

import numpy as np
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_sine_wav_b64(sr=16000, dur=0.2, freq=440.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Write WAV via simple PCM using soundfile indirectly through API runners
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode(
        "ascii"
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen2-Audio-7B-Instruct",
        "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    ],
)
def test_audio_text_to_text_contract_tuple_and_dict(model_id):
    audio_b64 = _make_sine_wav_b64()
    # Legacy tuple mode
    resp1 = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "audio",
            "inputs": {"audio_base64": audio_b64},
            "task": "audio-text-to-text",
            "options": {"_legacy_tuple": True, "_dummy_text": ""},
        },
    )
    assert resp1.status_code == 200, resp1.text
    # Dict mode
    resp2 = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "audio",
            "inputs": {"audio_base64": audio_b64},
            "task": "audio-text-to-text",
            "options": {"_legacy_tuple": False, "_dummy_text": ""},
        },
    )
    assert resp2.status_code == 200, resp2.text


@pytest.mark.parametrize(
    "model_id",
    [
        "nvidia/bigvgan_v2_22khz_80band_256x",
        "nvidia/bigvgan_v2_44khz_128band_512x",
    ],
)
def test_audio_to_audio_contract_tuple_and_dict(model_id):
    audio_b64 = _make_sine_wav_b64()
    # Legacy tuple mode
    resp1 = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "audio",
            "inputs": {"audio_base64": audio_b64},
            "task": "audio-to-audio",
            "options": {"_legacy_tuple": True},
        },
    )
    assert resp1.status_code == 200, resp1.text
    # Dict mode
    resp2 = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "audio",
            "inputs": {"audio_base64": audio_b64},
            "task": "audio-to-audio",
            "options": {"_legacy_tuple": False},
        },
    )
    assert resp2.status_code == 200, resp2.text


@pytest.mark.parametrize(
    "model_id",
    [
        "pyannote/segmentation-3.0",
        "pyannote/segmentation",
    ],
)
def test_vad_contract_tuple_and_dict(model_id):
    audio_b64 = _make_sine_wav_b64()
    # Legacy tuple mode
    resp1 = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "audio",
            "inputs": {"audio_base64": audio_b64},
            "task": "voice-activity-detection",
            "options": {"_legacy_tuple": True},
        },
    )
    assert resp1.status_code == 200, resp1.text
    # Dict mode
    resp2 = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "audio",
            "inputs": {"audio_base64": audio_b64},
            "task": "voice-activity-detection",
            "options": {"_legacy_tuple": False},
        },
    )
    assert resp2.status_code == 200, resp2.text
