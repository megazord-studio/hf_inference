"""Phase E 3D generation integration tests hitting /api/inference.
"""
from __future__ import annotations
from fastapi.testclient import TestClient
from app.main import app
import base64, io
from PIL import Image

client = TestClient(app)

HF_TEXT_3D_MODEL = "microsoft/TRELLIS-text-xlarge"
HF_IMAGE_3D_MODEL = "microsoft/TRELLIS-image-large"


def _mk_image_b64(color=(10, 120, 200)):
    img = Image.new("RGB", (32, 32), color=color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(task: str, model_id: str, inputs: dict, options: dict):
    resp = client.post('/api/inference', json={
        'model_id': model_id,
        'intent_id': '',
        'input_type': 'image' if 'image_base64' in inputs else 'text',
        'inputs': inputs,
        'task': task,
        'options': options,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()['result']


def test_image_to_3d_hf_model():
    result = _post('image-to-3d', HF_IMAGE_3D_MODEL, {'image_base64': _mk_image_b64()}, {})
    assert 'error' not in result, f"unexpected error: {result.get('error')}"
    # For now we only assert that a task_output dict is present; shape will be tightened once model contract is clear
    assert isinstance(result.get('task_output'), dict)


def test_text_to_3d_hf_model():
    result = _post('text-to-3d', HF_TEXT_3D_MODEL, {'text': 'a small cube'}, {})
    assert 'error' not in result, f"unexpected error: {result.get('error')}"
    out = result['task_output']
    assert isinstance(out.get('text'), str)
    assert len(out['text']) > 0

