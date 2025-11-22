"""Phase B vision generation integration tests hitting /api/inference.

Uses small prompts and placeholder behaviors for super-resolution & restoration.
Text-to-image and image-to-image require a diffusers model ID; kept configurable via simple default.
"""
from __future__ import annotations
from fastapi.testclient import TestClient
from app.main import app
import base64, io
from PIL import Image
import pytest

client = TestClient(app)

# Select baseline SD model (can be changed later to a smaller variant if available)
SD_MODEL = "ehristoforu/stable-diffusion-v1-5-tiny"

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
    result = resp.json()['result']
    assert 'error' not in result, f"unexpected error: {result.get('error')}"
    return result['task_output']


def test_text_to_image_generation_minimal():
    out = _post('text-to-image', SD_MODEL, {'text': 'a red cube on a table'}, {'num_inference_steps': 3, 'guidance_scale': 5.0})
    assert isinstance(out, dict)


def test_image_to_image_transformation_minimal():
    base = _mk_image_b64()
    out = _post('image-to-image', SD_MODEL, {'image_base64': base, 'text': 'make it brighter'}, {'strength': 0.5, 'num_inference_steps': 3})
    assert isinstance(out, dict)


def test_image_super_resolution_placeholder():
    base = _mk_image_b64()
    out = _post('image-super-resolution', 'placeholder/sr', {'image_base64': base}, {'scale': 2})
    assert isinstance(out, dict)


def test_image_restoration_placeholder():
    base = _mk_image_b64()
    out = _post('image-restoration', 'placeholder/restoration', {'image_base64': base}, {})
    assert isinstance(out, dict)
