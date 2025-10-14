import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("facebook/wav2vec2-base-960h", {"audio_path": "audio.wav"}),
        ("openai/whisper-base", {"audio_path": "audio.wav"}),
        (
            "facebook/wav2vec2-large-960h-lv60-self",
            {"audio_path": "audio.wav"},
        ),
        (
            "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            {"audio_path": "audio.wav"},
        ),
    ],
)
def test_automatic_speech_recognition(client, sample_audio, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="automatic-speech-recognition", payload={}
    )
    files = {"audio": ("test.wav", sample_audio, "audio/wav")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
