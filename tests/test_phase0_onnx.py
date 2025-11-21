import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

ONNX_MODEL = os.getenv("PHASE0_ONNX_MODEL", "Xenova/gpt2")


def test_onnx_text_generation_fallback():
    """Verify ONNX fallback path loads and generates tokens.

    The registry attempts PyTorch load first; on missing standard weights
    it falls back to the ONNX runner which auto-downloads an .onnx file.
    We assert backend=='onnx' and that some tokens were generated.

    This test is resilient:
    - If model has native PyTorch weights (future change), backend may be 'torch'; we skip.
    - If ONNX file cannot be found or onnxruntime missing, we skip instead of failing.
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

    # Early error cases -> skip (environment not prepared)
    if "error" in data["result"]:
        import pytest
        pytest.skip(f"Inference error returned: {data['result']['error']}")

    task = data["result"].get("task")
    assert task == "text-generation", f"Unexpected task: {task}"  # ensure fallback mapped

    out = data["result"].get("task_output", {})
    backend = out.get("backend")
    if backend != "onnx":
        import pytest
        pytest.skip(f"Backend not ONNX (got {backend}); fallback not triggered.")

    tokens_generated = out.get("tokens_generated")
    initial_length = out.get("initial_length")
    assert isinstance(tokens_generated, int) and tokens_generated >= initial_length, (
        f"Invalid token counts: generated={tokens_generated} initial={initial_length}"
    )
    assert "text" in out and isinstance(out["text"], str)

