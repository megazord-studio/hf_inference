import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Skip tests requiring network if HF_TOKEN missing; we rely on public small models (gpt2, distilbert, sentence-transformers) which are public.
GEN_MODEL = os.getenv("PHASE0_GEN_MODEL", "gpt2")
CLS_MODEL = os.getenv("PHASE0_CLS_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
EMB_MODEL = os.getenv("PHASE0_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def test_text_generation_basic():
    resp = client.post("/api/inference", json={
        "model_id": GEN_MODEL,
        "input_type": "text",
        "inputs": {"text": "Hello world"},
        "task": "text-generation",
        "options": {"max_new_tokens": 5, "temperature": 0.0}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"].get("task") == "text-generation"
    assert "task_output" in data["result"]
    assert isinstance(data["result"]["task_output"].get("text"), str)


def test_text_classification_basic():
    resp = client.post("/api/inference", json={
        "model_id": CLS_MODEL,
        "input_type": "text",
        "inputs": {"text": "I love this movie"},
        "task": "text-classification",
        "options": {}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"].get("task") == "text-classification"
    labels = data["result"]["task_output"].get("labels")
    assert isinstance(labels, list)
    assert labels, "No labels returned"


def test_embeddings_basic():
    resp = client.post("/api/inference", json={
        "model_id": EMB_MODEL,
        "input_type": "text",
        "inputs": {"text": "Sentence for embedding"},
        "task": "embedding",
        "options": {}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"].get("task") == "embedding"
    emb = data["result"]["task_output"].get("embedding")
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert data["result"]["task_output"].get("dim") == len(emb)


def test_registry_status_and_unload():
    # Trigger a generation load
    resp = client.post("/api/inference", json={
        "model_id": GEN_MODEL,
        "input_type": "text",
        "inputs": {"text": "Ping"},
        "task": "text-generation",
        "options": {"max_new_tokens": 2, "temperature": 0.0}
    })
    assert resp.status_code == 200
    status = client.get("/api/models/status").json()
    assert any(GEN_MODEL in k for k in status["loaded"].keys())
    # Unload
    del_resp = client.delete(f"/api/models/text-generation/{GEN_MODEL}")
    assert del_resp.status_code == 200
    status2 = client.get("/api/models/status").json()
    assert not any(GEN_MODEL in k for k in status2["loaded"].keys())
