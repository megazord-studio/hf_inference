import pytest

PARAMS = [
    "Xenova/gpt2",
]


@pytest.mark.parametrize("model_id", PARAMS)
def test_onnx_text_generation_fallback(client, model_id):
    payload = {
        "model_id": model_id,
        "input_type": "text",
        "inputs": {"text": "Hello ONNX"},
        "task": "text-generation",
        "options": {"max_new_tokens": 5, "temperature": 0.0}
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    result = data.get("result", {})
    assert result.get("task") == "text-generation"
    out = result.get("task_output", {})

    # Common assertions
    assert isinstance(out.get("text", ""), str)
    assert "backend" in out or result.get("backend")

    backend = out.get("backend") or result.get("backend")
    assert backend in {"onnx", "torch"}, f"Unexpected backend: {backend}"

    # Backend-specific assertions
    if backend == "onnx":
        assert out.get("approximate") is True
        assert isinstance(out.get("tokens_generated"), int) and out.get("tokens_generated") >= out.get("initial_length", 0)
        params = out.get("parameters", {})
        assert params.get("max_new_tokens") == 5
        assert params.get("temperature") == 0.0
    else:  # torch path
        generated = out.get("text", "")
        assert len(generated) >= len("Hello ONNX"), "Generated text shorter than input prompt"
