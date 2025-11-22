"""Phase C tests: zero-shot image classification, zero-shot object detection, keypoint detection.
All tests call inference endpoint (integration style).
"""
from __future__ import annotations
from fastapi.testclient import TestClient
from app.main import app
import base64, io
from PIL import Image

client = TestClient(app)

ZSI_MODEL = "openai/clip-vit-base-patch32"
ZSO_MODEL = "google/owlv2-base-patch16-ensemble"
KP_MODEL = "Xenova/yolov8n-pose"


def _mk_image_b64(color=(150, 60, 30)):
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(task: str, model_id: str, inputs: dict, options: dict):
    resp = client.post('/api/inference', json={
        'model_id': model_id,
        'intent_id': '',
        'input_type': 'image',
        'inputs': inputs,
        'task': task,
        'options': options,
    })
    assert resp.status_code == 200, resp.text
    result = resp.json()['result']
    assert 'error' not in result, f"unexpected error: {result.get('error')}"
    return result['task_output']

def test_zero_shot_image_classification_basic():
    out = _post('zero-shot-image-classification', ZSI_MODEL, {'image_base64': _mk_image_b64(), 'candidate_labels': ['red', 'brown', 'blue']}, {})
    assert 'labels' in out and 'scores' in out
    assert len(out['labels']) == len(out['scores'])

def test_zero_shot_object_detection_basic():
    out = _post('zero-shot-object-detection', ZSO_MODEL, {'image_base64': _mk_image_b64(), 'candidate_labels': ['person', 'dog']}, {})
    assert isinstance(out, dict)

def test_keypoint_detection_basic():
    out = _post('keypoint-detection', KP_MODEL, {'image_base64': _mk_image_b64()}, {})
    assert 'keypoints' in out and 'count' in out
    assert isinstance(out['keypoints'], list)
