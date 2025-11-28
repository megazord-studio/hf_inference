import pytest

PARAMS = [
    (
        "google/vit-base-patch16-224",
        "image-classification",
        "image",
        {"image_base64": None},
        "predictions",
    ),
    (
        "superb/wav2vec2-base-superb-ks",
        "audio-classification",
        "audio",
        {"audio_base64": None},
        "predictions",
    ),
    (
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "image-segmentation",
        "image",
        {"image_base64": None},
        "labels",
    ),
    (
        "hustvl/yolos-tiny",
        "object-detection",
        "image",
        {"image_base64": None},
        "detections",
    ),
    (
        "Intel/dpt-hybrid-midas",
        "depth-estimation",
        "image",
        {"image_base64": None},
        "depth_summary",
    ),
]


def _make_image_b64(color=(255, 255, 255), size=(32, 32)):
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_silent_wav_b64(duration_sec=0.5, sr=16000):
    import base64
    import io
    import wave

    import numpy as np

    samples = int(sr * duration_sec)
    data = np.zeros(samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


@pytest.mark.parametrize("model_id,task,input_type,inputs,expected", PARAMS)
def test_vision_audio_parametrized(
    client, model_id, task, input_type, inputs, expected
):
    if input_type == "image" and inputs.get("image_base64") is None:
        inputs["image_base64"] = _make_image_b64()
    if input_type == "audio" and inputs.get("audio_base64") is None:
        inputs["audio_base64"] = _make_silent_wav_b64()

    resp = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": input_type,
            "inputs": inputs,
            "task": task,
            "options": {},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    out = data["task_output"]
    assert expected in out or any(k == expected for k in out.keys())
