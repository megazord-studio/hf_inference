

def test_feature_extraction_alias_maps_to_embedding(client):
    payload = {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": "Embedding test sentence"},
        "task": "feature-extraction",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "embedding"
    out = data.get("task_output", {})
    assert (
        isinstance(out.get("embedding"), list)
        and len(out.get("embedding")) > 0
    )
