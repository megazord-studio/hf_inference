import pytest


def _make_image_b64(color=(50, 120, 210), size=(32, 32)):
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


@pytest.mark.parametrize(
    "model_id",
    [
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        "black-forest-labs/FLUX.1-Kontext-dev",
    ],
)
def test_image_to_image_api_mismatch_models(client, model_id):
    img_b64 = _make_image_b64()
    resp = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "image",
            "inputs": {"image_base64": img_b64, "text": "a simple variant"},
            "task": "image-to-image",
            "options": {"num_inference_steps": 2, "strength": 0.75},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert isinstance(data, dict)
    assert data.get("image_base64")


@pytest.mark.parametrize(
    "model_id",
    [
        "stabilityai/stable-video-diffusion-img2vid",
        "stabilityai/stable-video-diffusion-img2vid-xt",
    ],
)
def test_image_to_video_api_mismatch_models(client, model_id):
    img_b64 = _make_image_b64()
    resp = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "image",
            "inputs": {"image_base64": img_b64, "text": "short motion"},
            "task": "image-to-video",
            "options": {"num_frames": 2, "fps": 2},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert isinstance(data, dict)
    assert data.get("video_base64")
