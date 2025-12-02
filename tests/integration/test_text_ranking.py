import pytest


def test_text_ranking_basic(client):
    model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    query = "What is the capital of France?"
    candidates = [
        "Berlin is the capital of Germany.",
        "Madrid is Spain's central capital.",
        "Paris is the capital and most populous city of France.",
    ]
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"query": query, "candidates": candidates},
        "task": "text-ranking",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "text-ranking"
    out = data.get("task_output", {})
    scores = out.get("scores")
    indices = out.get("indices")
    assert isinstance(scores, list) and len(scores) == len(candidates)
    assert isinstance(indices, list) and sorted(indices) == list(range(len(candidates)))
