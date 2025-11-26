from __future__ import annotations
import base64, io
from PIL import Image
import pytest

PARAMS = [
    ("openai/clip-vit-base-patch32", 'zero-shot-image-classification', 'image', {'image_base64': None, 'candidate_labels': ['red','brown','blue']}, 'labels'),
    ("google/owlv2-base-patch16-ensemble", 'zero-shot-object-detection', 'image', {'image_base64': None, 'candidate_labels': ['person','dog']}, 'detections'),
    ("Xenova/yolov8n-pose", 'keypoint-detection', 'image', {'image_base64': None}, 'keypoints'),
]


def _mk_image_b64(color=(150, 60, 30)):
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("model_id,task,input_type,inputs,expected", PARAMS)
def test_vision_understanding_parametrized(client, model_id, task, input_type, inputs, expected):
    if inputs.get('image_base64') is None:
        inputs['image_base64'] = _mk_image_b64()
    resp = client.post('/api/inference', json={
        'model_id': model_id,
        'intent_id': '',
        'input_type': input_type,
        'inputs': inputs,
        'task': task,
        'options': {},
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()['result']
    assert 'error' not in data
    out = data['task_output']
    assert expected in out or any(k == expected for k in out.keys())
