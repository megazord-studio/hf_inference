def test_translation_basic(client):
    # T5-small is lightweight and suitable for tests
    model_id = "google-t5/t5-small"
    text = "Hello world"
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "text",
        "inputs": {"text": text},
        "task": "translation",
        "options": {
            "src_lang": "English",
            "tgt_lang": "French",
            "max_new_tokens": 20,
        },
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "translation"
    out = data.get("task_output", {})
    assert isinstance(out.get("text"), str)
    assert len(out.get("text", "")) > 0
