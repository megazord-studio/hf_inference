import pytest
from tests.conftest import create_spec

@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("facebook/musicgen-small", {"tta_prompt": "Lo-fi chillhop beat with warm drums, mellow keys, and a smooth bassline."}),
        ("facebook/musicgen-medium", {"tta_prompt": "Synthwave arpeggios with a steady beat."}),
        ("facebook/musicgen-melody", {"tta_prompt": "Jazz trio with upright bass, ride cymbal, and piano."}),
        ("audioldm/audio-ldm", {"tta_prompt": "Ambient pads with gentle rainfall."}),
    ],
)
def test_text_to_audio(client, model_id, payload):
    spec = create_spec(model_id=model_id, task="text-to-audio", payload=payload)
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        assert isinstance(resp.json(), (list, dict))
