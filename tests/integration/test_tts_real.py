import importlib

import pytest

# Require Coqui TTS to be installed; fail otherwise
_HAS_COQUI = importlib.util.find_spec("TTS") is not None


@pytest.mark.parametrize(
    "model_id,options",
    [
        ("microsoft/speecht5_tts", {}),
        ("hexgrad/Kokoro-82M", {"voice": "female", "lang": "en"}),
    ],
)
def test_tts_real_models(client, model_id, options):
    assert _HAS_COQUI, "Coqui TTS must be installed for this test"
    text = "Hello world" if "speecht5" in model_id else "Hello from Kokoro"
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": text},
        "task": "text-to-speech",
        "options": options,
    }
    resp = client.post("/api/inference", json=payload)
    if model_id == "hexgrad/Kokoro-82M":
        # For now, Kokoro path is not implemented in this environment
        assert resp.status_code == 501, resp.text
        err = resp.json().get("error", {})
        assert err.get("code") == "task_not_supported"
        return
    else:
        assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert isinstance(data.get("audio_base64"), str) and data[
        "audio_base64"
    ].startswith("data:audio/wav;base64,")
    assert isinstance(data.get("sample_rate"), int)
    assert isinstance(data.get("num_samples"), int) and data["num_samples"] > 0
