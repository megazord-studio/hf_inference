import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_id,task,input_type,inputs",
    [
        ("facebook/bart-large-cnn", "summarization", "text", {"text": "This is a long text to summarize."}),
        ("openai/whisper-tiny", "automatic-speech-recognition", "audio", {"audio_base64": None}),
    ],
)
def test_parametrized_minimal_inference(client, model_id, task, input_type, inputs):
    if input_type == "image" and inputs.get("image_base64") is None:
        pytest.skip("Image input not provided; placeholder param")
    if input_type == "audio" and inputs.get("audio_base64") is None:
        pytest.skip("Audio input not provided; placeholder param")

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
    data = resp.json()
    assert "result" in data

