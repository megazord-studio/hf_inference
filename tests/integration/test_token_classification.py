import pytest


def test_token_classification_basic(client):
    payload = {
        "model_id": "dslim/bert-base-NER",
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": "Hugging Face Inc. is a company based in New York City."},
        "task": "token-classification",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "token-classification"
    out = data.get("task_output", {})
    ents = out.get("entities")
    assert isinstance(ents, list)
    if ents:
        e0 = ents[0]
        for k in ("entity", "score", "start", "end"):
            assert k in e0
