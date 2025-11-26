from app.core.utils.media import (
    encode_image_base64,
)
from PIL import Image
import pytest

PARAMS = [
    ("ali-vilab/text-to-video-ms-1.7b", 'text-to-video', 'text', {'text': 'a tiny moving square', 'options': {'num_frames': 4}}, 'video_base64'),
    ("ali-vilab/text-to-video-ms-1.7b", 'image-to-video', 'video', {'image_base64': None, 'text': 'make the object move', 'options': {'num_frames': 4}}, 'video_base64'),
]


def _make_image_b64(size=(64, 64)) -> str:
    img = Image.new("RGB", size, (128, 64, 32))
    return encode_image_base64(img)


@pytest.mark.parametrize("model_id,task,input_type,inputs,expected", PARAMS)
def test_video_generation_parametrized(client, model_id, task, input_type, inputs, expected):
    if input_type == 'video' and inputs.get('image_base64') is None:
        inputs['image_base64'] = _make_image_b64()

    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": input_type,
        "inputs": {k: v for k, v in inputs.items() if k != 'options'},
        "task": task,
        "options": inputs.get('options', {}),
    }
    resp = client.post("/api/inference", json=payload, params={"include_model_meta": False})
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    out = data.get("task_output", {})
    assert expected in out
