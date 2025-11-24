from fastapi.testclient import TestClient
from app.main import app

from app.core.utils.media import (
    encode_image_base64,
)
from PIL import Image
import base64

client = TestClient(app)


TEXT_TO_VIDEO_MODEL_ID = "ali-vilab/text-to-video-ms-1.7b"


def _make_image_b64(size=(64, 64)) -> str:
    img = Image.new("RGB", size, (128, 64, 32))
    return encode_image_base64(img)


def test_api_inference_text_to_video_real_model():
    payload = {
        "model_id": TEXT_TO_VIDEO_MODEL_ID,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": "a tiny video of a moving square"},
        "task": "text-to-video",
        "options": {"num_frames": 4, "height": 256, "width": 256, "fps": 4},
    }
    resp = client.post("/api/inference", json=payload, params={"include_model_meta": False})
    assert resp.status_code == 200
    data = resp.json()
    result = data["result"]
    assert result.get("task") == "text-to-video"
    out = result.get("task_output")
    assert isinstance(out, dict)
    vb64 = out.get("video_base64")
    assert isinstance(vb64, str)
    assert vb64.startswith("data:video/mp4;base64,")
    assert out.get("num_frames") == 4
    assert "error" not in result


def test_api_inference_image_to_video_real_model():
    img_b64 = _make_image_b64()
    payload = {
        "model_id": TEXT_TO_VIDEO_MODEL_ID,
        "intent_id": None,
        "input_type": "video",
        "inputs": {"image_base64": img_b64, "text": "make the object move"},
        "task": "image-to-video",
        "options": {"num_frames": 4, "fps": 4},
    }
    resp = client.post("/api/inference", json=payload, params={"include_model_meta": False})
    assert resp.status_code == 200
    data = resp.json()
    result = data["result"]
    assert result.get("task") == "image-to-video"
    out = result.get("task_output")
    assert isinstance(out, dict)
    vb64 = out.get("video_base64")
    assert isinstance(vb64, str)
    assert vb64.startswith("data:video/mp4;base64,")
    assert out.get("num_frames") == 4
    assert "error" not in result
