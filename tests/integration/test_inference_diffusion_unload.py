import time

import pytest

PARAMS = [
    "runwayml/stable-diffusion-v1-5",
    "ehristoforu/stable-diffusion-v1-5-tiny",
]


@pytest.mark.integration
@pytest.mark.parametrize("model_id", PARAMS)
def test_diffusion_unload_regression_heavy_model_eviction(client, model_id):
    """Regression: heavy diffusion models should be eligible for unload after use.

    Uses the shared `client` fixture from `conftest.py` and validates that
    the `/api/models/status` endpoint contains Phase B metadata for the
    heavy diffusion model used in other vision tests.
    """
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"prompt": "a tiny cat"},
        "task": "text-to-image",
        "options": {"num_inference_steps": 4},
    }

    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text

    # Allow a brief cooldown for any async unload scheduler
    time.sleep(0.5)

    status_resp = client.get("/api/models/status")
    assert status_resp.status_code == 200, status_resp.text
    status = status_resp.json()

    # Expect the heavy model to be present in status, with fields tracked by Phase B
    key = f"{model_id}:text-to-image"
    loaded = status.get("loaded", {})
    assert key in loaded, f"expected {key} in loaded map"
    info = loaded[key]
    assert info.get("status") in ("ready", "error"), info
    assert isinstance(info.get("load_time_ms"), int), info

    # Regression check: ensure unload/eviction metadata is present
    assert "mem_mb" in info or "evict_weight" in info
