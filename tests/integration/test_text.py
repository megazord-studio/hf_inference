import pytest

PARAMS = [
    ("gpt2", "text-generation", "text", {"text": "Hello world"}, "text"),
    ("distilbert-base-uncased-finetuned-sst-2-english", "text-classification", "text", {"text": "I love this movie"}, "labels"),
    ("sentence-transformers/all-MiniLM-L6-v2", "embedding", "text", {"text": "Sentence for embedding"}, "embedding"),
]


@pytest.mark.parametrize("model_id,task,input_type,inputs,expected", PARAMS)
def test_text_inference_parametrized(client, model_id, task, input_type, inputs, expected):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": input_type,
        "inputs": inputs,
        "task": task,
        "options": {},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == task
    out = data.get("task_output", {})
    # Basic shape assertion
    if expected == "embedding":
        assert isinstance(out.get("embedding"), list)
    else:
        assert expected in out or any(k == expected for k in out.keys()), f"Expected field {expected} in output keys: {list(out.keys())}"
