import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "facebook/mms-tts-eng",
            {"tts_text": "Hello from Switzerland! This is a TTS demo."},
        ),
        ("facebook/mms-tts-deu", {"tts_text": "Guten Tag aus der Schweiz!"}),
    ],
)
def test_text_to_speech(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="text-to-speech", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        # Check if response is binary (audio file) or JSON
        content_type = resp.headers.get("content-type", "")
        if (
            "audio" in content_type
            or "application/octet-stream" in content_type
        ):
            # Binary response - verify it's not empty
            assert len(resp.content) > 0
        else:
            # JSON response
            data = resp.json()
            assert isinstance(data, (list, dict))
            check_response_for_skip_or_error(data, model_id)
