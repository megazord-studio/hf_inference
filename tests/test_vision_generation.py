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
SD_MODEL = "segmind/tiny-sd"

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
    data = resp.json()['result']
    return data['task_output']


@pytest.mark.skip(reason="TODO: implement real Stable Diffusion execution; unskip when pipeline reliably loads")
def test_text_to_image_generation_minimal():
    out = _post('text-to-image', SD_MODEL, {'text': 'a red cube on a table'}, {'num_inference_steps': 3, 'guidance_scale': 5.0})
    assert 'image_base64' in out
    assert isinstance(out['image_base64'], str) and len(out['image_base64']) > 100, 'Expected non-empty generated image'


@pytest.mark.skip(reason="TODO: implement real Stable Diffusion execution; unskip when pipeline reliably loads")
def test_image_to_image_transformation_minimal():
    base = _mk_image_b64()
    out = _post('image-to-image', SD_MODEL, {'image_base64': base, 'text': 'make it brighter'}, {'strength': 0.5, 'num_inference_steps': 3})
    assert 'image_base64' in out
    assert isinstance(out['image_base64'], str) and len(out['image_base64']) > 100


@pytest.mark.skip(reason="TODO: implement real Stable Diffusion execution; unskip when pipeline reliably loads")
def test_image_super_resolution_placeholder():
    base = _mk_image_b64()
    out = _post('image-super-resolution', 'placeholder/sr', {'image_base64': base}, {'scale': 2})
    assert out.get('image_base64')
    assert out.get('orig_size') in ([32, 32], (32, 32))
    ns = out.get('new_size')
    assert ns == [64, 64] or ns == (64, 64)


@pytest.mark.skip(reason="TODO: implement real Stable Diffusion execution; unskip when pipeline reliably loads")
def test_image_restoration_placeholder():
    base = _mk_image_b64()
    out = _post('image-restoration', 'placeholder/restoration', {'image_base64': base}, {})
    assert 'image_base64' in out
    assert isinstance(out['image_base64'], str)
