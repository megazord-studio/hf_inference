import pytest

PARAMS = [
    "stabilityai/stable-diffusion-2-1",
]


@pytest.mark.integration
@pytest.mark.parametrize("model_id", PARAMS)
def test_models_status_includes_loaded_heavy_model(client, model_id):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"prompt": "a small test image"},
        "task": "text-to-image",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text

    status_resp = client.get("/api/models/status")
    assert status_resp.status_code == 200, status_resp.text
    data = status_resp.json()
    assert "loaded" in data
    loaded = data["loaded"]

    key = f"{payload['model_id']}:{payload['task']}"
    assert key in loaded, f"expected key {key} in {loaded.keys()}"
    info = loaded[key]

    # Fields that Phase B guarantees
    assert info["status"] in ("ready", "error")
    assert isinstance(info.get("load_time_ms"), int)
    mem_mb = info.get("mem_mb")
    if mem_mb is not None:
        assert mem_mb > 0
