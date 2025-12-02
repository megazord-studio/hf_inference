import pytest


def _make_image_b64(color=(160, 160, 160), size=(64, 48)):
    import base64
    import io
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a few shapes to produce corners
    draw.rectangle([8, 8, size[0]-8, size[1]-8], outline=(0,0,0), width=2)
    draw.line([8, size[1]//2, size[0]-8, size[1]//2], fill=(0,0,0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


@pytest.mark.parametrize(
    "model_id",
    [
        "ETH-CVG/lightglue_superpoint",
        "usyd-community/vitpose-plus-base",
    ],
)
def test_keypoint_detection_compat_models(client, model_id):
    img_b64 = _make_image_b64()
    resp = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "image",
            "inputs": {"image_base64": img_b64},
            "task": "keypoint-detection",
            "options": {"top_k": 32},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert isinstance(data, dict)
    assert "keypoints" in data and isinstance(data["keypoints"], list)
    # At least a few corners should be found
    assert data.get("count", 0) >= 4
