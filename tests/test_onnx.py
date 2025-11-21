import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

ONNX_MODEL = os.getenv("PHASE0_ONNX_MODEL", "Xenova/gpt2")


def test_onnx_text_generation_fallback():
    """Verify text generation works and returns expected fields.

    Original version skipped when backend wasn't ONNX or on early errors.
    This version asserts correctness for either torch or onnx backends so the
    test is never skipped.
    """
    payload = {
        "model_id": ONNX_MODEL,
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
        # Minimal guarantees: generation produced non-empty text or at least input echoed with extra tokens
        generated = out.get("text", "")
        assert len(generated) >= len("Hello ONNX"), "Generated text shorter than input prompt"
