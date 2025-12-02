import pytest


@pytest.mark.parametrize(
    "model_id,question,table,expected_substring",
    [
        (
            "google/tapas-base-finetuned-wtq",
            "What is the age of Alice?",
            [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]],
            "30",
        ),
    ],
)
def test_table_question_answering_basic(client, model_id, question, table, expected_substring):
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"question": question, "table": table},
        "task": "table-question-answering",
        "options": {"topk": 1},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "table-question-answering"
    out = data.get("task_output", {})
    assert isinstance(out.get("answer"), str)
    # The model should return the cell value '30' for Alice's age
    assert expected_substring in out.get("answer", "") or expected_substring in "".join(out.get("cells", []))
