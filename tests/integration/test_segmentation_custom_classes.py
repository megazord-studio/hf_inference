import base64
import io
from PIL import Image
import pytest

from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def _load_asset_b64():
    # Use existing test asset
    p = 'tests/assets/image.jpg'
    with open(p, 'rb') as f:
        data = f.read()
    return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('utf-8')


MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def test_segmentation_no_classes():
    img_b64 = _load_asset_b64()
    payload = {
        "task": "image-segmentation",
        "model_id": MODEL_ID,
        "input_type": "image",
        "inputs": {"image_base64": img_b64},
        "options": {},
    }
    r = client.post("/api/inference", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    task_output = data.get("result", {}).get("task_output", {})
    assert isinstance(task_output.get("labels"), dict)
    assert isinstance(task_output.get("shape"), list)
    assert len(task_output["shape"]) == 2


def test_segmentation_filter_classes_list():
    img_b64 = _load_asset_b64()
    req_classes = ["person", "sky", "wall", "car"]
    payload = {
        "task": "image-segmentation",
        "model_id": MODEL_ID,
        "input_type": "image",
        "inputs": {"image_base64": img_b64},
        "options": {"classes": req_classes},
    }
    r = client.post("/api/inference", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    task_output = data.get("result", {}).get("task_output", {})
    labels = task_output.get("labels", {})
    # All returned labels must be within requested set (case-insensitive)
    req_lower = {c.lower() for c in req_classes}
    for k in labels.keys():
        assert k.lower() in req_lower


def test_segmentation_rename_classes_mapping_no_match_errors():
    img_b64 = _load_asset_b64()
    # Use unlikely class names to trigger no-match error
    mapping = {"unicorn": "myth"}
    payload = {
        "task": "image-segmentation",
        "model_id": MODEL_ID,
        "input_type": "image",
        "inputs": {"image_base64": img_b64},
        "options": {"classes": mapping},
    }
    r = client.post("/api/inference", json=payload)
    assert r.status_code == 500, r.text
    data = r.json()
    assert data.get("error", {}).get("code") == "inference_failed"
    # Message should include our specific error hint
    assert "segmentation_no_class_match" in data.get("error", {}).get("message", "")


def test_segmentation_rename_classes_mapping_success_if_present():
    img_b64 = _load_asset_b64()
    # Map common ADE20K class names to aliases. If present in image, they will be renamed.
    mapping = {"person": "human", "car": "vehicle"}
    payload = {
        "task": "image-segmentation",
        "model_id": MODEL_ID,
        "input_type": "image",
        "inputs": {"image_base64": img_b64},
        "options": {"classes": mapping},
    }
    r = client.post("/api/inference", json=payload)
    # Either 200 with filtered labels OR 500 if none matched; both reflect correct behavior
    assert r.status_code in (200, 500), r.text
    if r.status_code == 200:
        data = r.json()
        task_output = data.get("result", {}).get("task_output", {})
        labels = task_output.get("labels", {})
        # If labels exist, they must be aliased names only
        for k in labels.keys():
            assert k in set(mapping.values())
    else:
        data = r.json()
        assert data.get("error", {}).get("code") == "inference_failed"
        assert "segmentation_no_class_match" in data.get("error", {}).get("message", "")
