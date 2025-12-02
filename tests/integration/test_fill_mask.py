import pytest


@pytest.mark.parametrize(
    "model_id,text",
    [
        (
            "distilbert/distilbert-base-uncased",
            "Paris is the [MASK] of France.",
        ),
        ("google-bert/bert-base-uncased", "The capital of Japan is [MASK]."),
    ],
)
def test_fill_mask_basic(client, model_id, text):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": text},
        "task": "fill-mask",
        "options": {"top_k": 3},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "fill-mask"
    out = data.get("task_output", {})
    preds = out.get("predictions")
    assert isinstance(preds, list) and len(preds) > 0
    assert "label" in preds[0] and "score" in preds[0]
