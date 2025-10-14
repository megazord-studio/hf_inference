import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


def _clean(p):
    return {k: v for k, v in p.items() if not k.endswith("_path")}


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "laion/clap-htsat-fused",
            {
                "audio_path": "audio.wav",
                "candidate_labels": ["speech", "music", "rain", "applause"],
            },
        ),
        (
            "laion/clap-htsat-unfused",
            {
                "audio_path": "audio.wav",
                "candidate_labels": ["guitar", "violin", "drums"],
            },
        ),
    ],
)
def test_zero_shot_audio_classification(
    client, sample_audio, model_id, payload
):
    spec = create_spec(
        model_id=model_id,
        task="zero-shot-audio-classification",
        payload=_clean(payload),
    )
    files = {"audio": ("test.wav", sample_audio, "audio/wav")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
