from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

MM_MODELS = [
    "Salesforce/blip-vqa-base",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen-VL-Chat",
    "google/gemma-3-1b-it",
    "HuggingFaceM4/idefics2-8b",
    "openbmb/MiniCPM-Llama3-V-2_5",
    "01-ai/Yi-VL-6B",
    "OpenGVLab/InternVL2-8B",
    "microsoft/kosmos-2-patch14-224",
    "microsoft/Florence-2-base-ft",
    "google/paligemma-3b-pt-224",
    "THUDM/cogvlm2-llama3-chat-19B",
]

# Models that require HF authentication (gated models)
GATED_MODELS = {
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen-VL-Chat",
    "HuggingFaceM4/idefics2-8b",
    "openbmb/MiniCPM-Llama3-V-2_5",
    "01-ai/Yi-VL-6B",
    "OpenGVLab/InternVL2-8B",
    "google/paligemma-3b-pt-224",
    "THUDM/cogvlm2-llama3-chat-19B",
}

ANSWER_MODELS = {"Salesforce/blip-vqa-base"}


def _mk_image_b64(color=(200, 180, 50)):
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(client, task: str, model_id: str, inputs: dict, options: dict):
    # removed interactive debugger to allow automated test runs
    resp = client.post("/api/inference", json={
        "model_id": model_id,
        "intent_id": "",
        "input_type": "image",
        "inputs": inputs,
        "task": task,
        "options": options,
    })
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert "error" not in result, f"unexpected error: {result.get('error')}"
    return result["task_output"]

@pytest.mark.parametrize("model_id", MM_MODELS)
def test_image_text_to_text_basic(client, model_id: str):
    if model_id in GATED_MODELS:
        pytest.skip(f"Skipping gated model {model_id} - requires HF authentication")
    out = _post(client, "image-text-to-text", model_id, {"image_base64": _mk_image_b64(), "text": "What color is the square?"}, {"max_length": 10})
    if model_id in ANSWER_MODELS:
        assert isinstance(out.get("answer"), str)
        assert out.get("answer") != ""
        assert isinstance(out.get("arch"), str)
    else:
        assert isinstance(out, dict)
