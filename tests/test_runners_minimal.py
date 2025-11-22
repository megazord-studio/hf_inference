"""Phase 2 runners exercising with real public models via API integration tests.

All tests invoke the /api/inference endpoint using FastAPI's TestClient
instead of instantiating runner classes directly. This ensures routing,
request/response schemas, registry loading, and model lifecycle are
exercised end-to-end.

Note: These tests can be slow on the first run due to model downloads.
"""
from __future__ import annotations
import math, wave, io, base64
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Public small-ish models
CLS_MODEL = "google/vit-base-patch16-224"
CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
DET_MODEL = "hustvl/yolos-tiny"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
DEPTH_MODEL = "Intel/dpt-hybrid-midas"
ASR_MODEL = "facebook/wav2vec2-base-960h"
AUDIO_CLS_MODEL = "superb/wav2vec2-base-superb-ks"
TTS_MODEL = "microsoft/speecht5_tts"

# Helpers

def make_image_b64(size=(32, 32), color=(120, 34, 56)):
    from PIL import Image
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def make_audio_b64(duration_s: float = 0.25, sr: int = 16000):
    samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, samples, endpoint=False)
    wav_f = (0.1 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
    pcm = (wav_f * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"

# Contract builder

def _post(task: str, model_id: str, input_type: str, inputs: dict, options: dict):
    resp = client.post('/api/inference', json={
        'model_id': model_id,
        'intent_id': '',
        'input_type': input_type,
        'inputs': inputs,
        'task': task,
        'options': options,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()['result']
    assert data.get('task') == task
    return data['task_output']

# Tests

def test_image_classification_runner_integration():
    out = _post('image-classification', CLS_MODEL, 'image', {'image_base64': make_image_b64()}, {'top_k': 2})
    assert isinstance(out.get('predictions'), list)


def test_image_captioning_runner_integration():
    out = _post('image-captioning', CAPTION_MODEL, 'image', {'image_base64': make_image_b64()}, {'max_new_tokens': 5})
    assert isinstance(out.get('text'), str)


def test_object_detection_runner_integration():
    out = _post('object-detection', DET_MODEL, 'image', {'image_base64': make_image_b64()}, {'confidence': 0.0, 'max_detections': 3})
    assert isinstance(out.get('detections'), list)


def test_image_segmentation_runner_integration():
    out = _post('image-segmentation', SEG_MODEL, 'image', {'image_base64': make_image_b64()}, {})
    assert set(out.keys()) == {'labels', 'shape'}
    assert isinstance(out['labels'], dict)


def test_depth_estimation_runner_integration():
    out = _post('depth-estimation', DEPTH_MODEL, 'image', {'image_base64': make_image_b64()}, {})
    ds = out['depth_summary']
    assert set(ds.keys()) == {'mean', 'min', 'max', 'shape', 'len'}
    assert ds['len'] >= 0


def test_asr_runner_integration():
    out = _post('automatic-speech-recognition', ASR_MODEL, 'audio', {'audio_base64': make_audio_b64()}, {})
    assert 'text' in out


def test_audio_classification_runner_integration():
    out = _post('audio-classification', AUDIO_CLS_MODEL, 'audio', {'audio_base64': make_audio_b64()}, {'top_k': 2})
    assert 'predictions' in out


def test_tts_runner_integration():
    out = _post('text-to-speech', TTS_MODEL, 'text', {'text': 'hello'}, {})
    assert isinstance(out.get('audio_base64'), str)
