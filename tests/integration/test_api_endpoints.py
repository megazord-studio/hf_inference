import time


def test_intents_endpoint_shape(client):
    r = client.get("/api/intents")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert any(i["id"] == "summarize" for i in data)
    sample = data[0]
    for key in ["id", "label", "description", "input_types", "hf_tasks"]:
        assert key in sample


def test_models_preloaded_basic(client, monkeypatch):
    # Patch fetch to avoid heavy network
    from app.routers import models as models_module
    def fake_fetch(limit: int):
        return [models_module.ModelSummary(id=f"model-{i}", pipeline_tag="summarization", likes=10+i, downloads=100+i) for i in range(limit)]
    monkeypatch.setattr(models_module, "_fetch_models", fake_fetch)
    r = client.get("/api/models/preloaded?limit=5&refresh=true")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5
    assert data[0]["pipeline_tag"] == "summarization"


def test_models_cache(client, monkeypatch):
    from app.routers import models as models_module
    calls = {"n": 0}
    def fake_fetch(limit: int):
        calls["n"] += 1
        return [models_module.ModelSummary(id="m", pipeline_tag="summarization")]
    monkeypatch.setattr(models_module, "_fetch_models", fake_fetch)
    # First call refreshes
    r1 = client.get("/api/models/preloaded?limit=1&refresh=true")
    assert r1.status_code == 200
    # Second call without refresh should not refetch
    r2 = client.get("/api/models/preloaded?limit=1")
    assert r2.status_code == 200
    assert calls["n"] == 1


def test_models_meta(client, monkeypatch):
    from app.routers import models as models_module
    monkeypatch.setattr(models_module, "_CACHE", [])
    monkeypatch.setattr(models_module, "_CACHE_TS", time.time())
    r = client.get("/api/models/preloaded/meta")
    assert r.status_code == 200
    meta = r.json()
    assert "cached" in meta and "ttl" in meta


def test_inference_unknown_task_returns_clear_error(client):
    payload = {
        "model_id": "bert-base-uncased",
        "intent_id": "",
        "input_type": "text",
        "inputs": {"text": "hello"},
        "task": "non-existing-task-xyz",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload, params={"include_model_meta": False})
    assert resp.status_code == 501
    data = resp.json()
    assert data["error"]["code"] == "task_not_supported"
    assert data["error"].get("details", {}).get("task") == "non-existing-task-xyz"
