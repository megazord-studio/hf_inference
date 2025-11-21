import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Public small models
IMAGE_CLS_MODEL = "google/vit-base-patch16-224"
AUDIO_CLS_MODEL = "superb/wav2vec2-base-superb-ks"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
DET_MODEL = "hustvl/yolos-tiny"
DEPTH_MODEL = "Intel/dpt-hybrid-midas"
CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
ASR_MODEL = "facebook/wav2vec2-base-960h"
TTS_MODEL = "microsoft/speecht5_tts"

# Utility: build base64 image
def _make_image_b64(color=(255,255,255), size=(32,32)):
    import base64, io
    from PIL import Image
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO(); img.save(buf, format='PNG'); b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# Utility: build base64 silent audio (PCM WAV)
def _make_silent_wav_b64(duration_sec=0.5, sr=16000):
    import numpy as np, base64, io, wave
    samples = int(sr * duration_sec)
    data = np.zeros(samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(data.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def test_image_classification_basic():
    image_b64 = _make_image_b64()
    resp = client.post('/api/inference', json={
        'model_id': IMAGE_CLS_MODEL,
        'intent_id': '',
        'input_type': 'image',
        'inputs': {'image_base64': image_b64},
        'task': 'image-classification',
        'options': {}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    preds = data['task_output'].get('predictions')
    assert isinstance(preds, list)


def test_audio_classification_placeholder():
    audio_b64 = _make_silent_wav_b64()
    resp = client.post('/api/inference', json={
        'model_id': AUDIO_CLS_MODEL,
        'intent_id': '',
        'input_type': 'audio',
        'inputs': {'audio_base64': audio_b64},
        'task': 'audio-classification',
        'options': {}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    preds = data['task_output'].get('predictions')
    assert isinstance(preds, list)


def test_image_segmentation_labels_shape():
    image_b64 = _make_image_b64(color=(0,128,255), size=(48,48))
    resp = client.post('/api/inference', json={
        'model_id': SEG_MODEL,
        'intent_id': '',
        'input_type': 'image',
        'inputs': {'image_base64': image_b64},
        'task': 'image-segmentation',
        'options': {}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    labels = data['task_output'].get('labels')
    shape = data['task_output'].get('shape')
    assert isinstance(labels, dict)
    assert isinstance(shape, list)


def test_object_detection_structure():
    image_b64 = _make_image_b64(color=(128,0,64), size=(64,64))
    resp = client.post('/api/inference', json={
        'model_id': DET_MODEL,
        'intent_id': '',
        'input_type': 'image',
        'inputs': {'image_base64': image_b64},
        'task': 'object-detection',
        'options': {'confidence': 0.0}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    dets = data['task_output'].get('detections')
    assert isinstance(dets, list)


def test_depth_estimation_summary():
    image_b64 = _make_image_b64(color=(10,10,10), size=(40,40))
    resp = client.post('/api/inference', json={
        'model_id': DEPTH_MODEL,
        'intent_id': '',
        'input_type': 'image',
        'inputs': {'image_base64': image_b64},
        'task': 'depth-estimation',
        'options': {}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    summary = data['task_output'].get('depth_summary')
    assert isinstance(summary, dict)
    assert 'mean' in summary and 'len' in summary


def test_image_captioning_optional():
    image_b64 = _make_image_b64(color=(0,128,255), size=(64,64))
    resp = client.post('/api/inference', json={
        'model_id': CAPTION_MODEL,
        'intent_id': '',
        'input_type': 'image',
        'inputs': {'image_base64': image_b64},
        'task': 'image-captioning',
        'options': {'max_new_tokens': 10}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    text = data['task_output'].get('text')
    assert isinstance(text, str)


def test_automatic_speech_recognition_structure():
    audio_b64 = _make_silent_wav_b64()
    resp = client.post('/api/inference', json={
        'model_id': ASR_MODEL,
        'intent_id': '',
        'input_type': 'audio',
        'inputs': {'audio_base64': audio_b64},
        'task': 'automatic-speech-recognition',
        'options': {}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    text = data['task_output'].get('text')
    assert isinstance(text, str)


def test_text_to_speech_structure():
    resp = client.post('/api/inference', json={
        'model_id': TTS_MODEL,
        'intent_id': '',
        'input_type': 'text',
        'inputs': {'text': 'Hello world'},
        'task': 'text-to-speech',
        'options': {}
    })
    assert resp.status_code == 200
    data = resp.json()['result']
    audio_b64 = data['task_output'].get('audio_base64')
    assert isinstance(audio_b64, str)
