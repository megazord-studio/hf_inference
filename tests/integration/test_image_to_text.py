import pytest

# Minimal 1x1 PNG (white) data URL
_ONE_PIXEL_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

@pytest.mark.parametrize(
    "model_id,image_b64",
    [
        ("nlpconnect/vit-gpt2-image-captioning", _ONE_PIXEL_PNG),
    ],
)
def test_image_to_text_captioning(client, model_id, image_b64):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "image",
        "inputs": {"image_base64": image_b64},
        "task": "image-to-text",
        "options": {"max_new_tokens": 8},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "image-to-text"
    out = data.get("task_output", {})
    assert isinstance(out.get("text"), str)
