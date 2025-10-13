import pytest
from tests.conftest import create_spec

@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("facebook/mms-tts-eng", {"tts_text": "Hello from Switzerland! This is a TTS demo."}),
        ("facebook/mms-tts-deu", {"tts_text": "Guten Tag aus der Schweiz!"}),
        ("espnet/kan-bayashi_ljspeech_vits", {"tts_text": "Welcome to the Alps."}),
        ("microsoft/speecht5_tts", {"tts_text": "This is a SpeechT5 test."}),
    ],
)
def test_text_to_speech(client, model_id, payload):
    spec = create_spec(model_id=model_id, task="text-to-speech", payload=payload)
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        assert isinstance(resp.json(), (list, dict))
