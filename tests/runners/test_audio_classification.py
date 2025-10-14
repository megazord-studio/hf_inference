import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("superb/hubert-base-superb-er", {"audio_path": "audio.wav"}),
        (
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            {"audio_path": "audio.wav"},
        ),
    ],
)
def test_audio_classification(client, sample_audio, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="audio-classification", payload={}
    )
    files = {"audio": ("test.wav", sample_audio, "audio/wav")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
