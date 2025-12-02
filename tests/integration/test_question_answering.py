import pytest


@pytest.mark.parametrize(
    "model_id,question,context",
    [
        (
            "distilbert/distilbert-base-cased-distilled-squad",
            "Who wrote the novel '1984'?",
            "George Orwell wrote the novel '1984' in 1949.",
        ),
        (
            "deepset/roberta-base-squad2",
            "What is the capital of France?",
            "Paris is the capital and most populous city of France.",
        ),
    ],
)
def test_question_answering_basic(client, model_id, question, context):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"question": question, "context": context},
        "task": "question-answering",
        "options": {"topk": 1},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "question-answering"
    out = data.get("task_output", {})
    assert isinstance(out.get("answer"), str)
    assert "score" in out
