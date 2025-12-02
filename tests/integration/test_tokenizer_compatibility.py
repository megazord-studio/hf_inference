import base64
import io

import pytest
from PIL import Image


def _mk_image_b64():
    img = Image.new("RGB", (64, 64), color=(20, 140, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize(
    "model_id,task,labels",
    [
        (
            "IDEA-Research/grounding-dino-base",
            "zero-shot-object-detection",
            ["person", "dog"],
        ),
    ],
)
def test_grounding_dino_tokenizer_fast_upgrade(client, model_id, task, labels):
    resp = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "image",
            "inputs": {
                "image_base64": _mk_image_b64(),
                "candidate_labels": labels,
            },
            "task": task,
            "options": {},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert "detections" in data
    assert isinstance(data["detections"], list)
