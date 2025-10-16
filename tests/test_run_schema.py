from typing import List


def test_run_schema_known_task(client):
    response = client.get("/run", params={"task": "text-generation"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["task"] == "text-generation"

    schema = payload["schema"]
    assert schema["category"] == "text"
    assert schema["label"] == "Text generation"

    input_names: List[str] = [field["name"] for field in schema.get("inputs", [])]
    assert "prompt" in input_names


def test_run_schema_invalid_task(client):
    response = client.get("/run", params={"task": "not-a-real-task"})
    assert response.status_code == 400

    payload = response.json()
    assert "error" in payload
    assert "supported" in payload
