import pytest


@pytest.mark.parametrize(
    "model_id,a,b",
    [
        ("sentence-transformers/all-MiniLM-L6-v2", "A quick brown fox", "A fast brown fox"),
        ("sentence-transformers/all-mpnet-base-v2", "The cat sits on the mat.", "A feline is on the rug."),
    ],
)
def test_sentence_similarity_basic(client, model_id, a, b):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": a, "text_pair": b},
        "task": "sentence-similarity",
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "sentence-similarity"
    out = data.get("task_output", {})
    score = out.get("score")
    assert isinstance(score, float) or isinstance(score, int)
    assert -1.0 <= float(score) <= 1.0
