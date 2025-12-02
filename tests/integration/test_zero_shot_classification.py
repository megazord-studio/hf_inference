import pytest


def test_zero_shot_classification_basic(client):
    model_id = "facebook/bart-large-mnli"
    text = "The Eiffel Tower is a famous landmark in Paris."
    labels = ["travel", "cooking", "sports"]
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": text, "labels": labels},
        "task": "zero-shot-classification",
        "options": {"multi_label": False},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "zero-shot-classification"
    out = data.get("task_output", {})
    preds = out.get("predictions")
    assert isinstance(preds, list) and len(preds) == len(labels)
    assert all("label" in p and "score" in p for p in preds)
