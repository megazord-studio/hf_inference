

def test_any_to_any_basic(client, infer):
    # Minimal mixed-modality payload; should not 501
    payload = infer(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        task="any-to-any",
        input_type="text",
        inputs={
            "text": "General test prompt across modalities.",
            "image_base64": "data:image/png;base64,iVBORw0KGgo=",
            "audio_base64": "data:audio/wav;base64,UklGRg==",
            "video_base64": "data:video/mp4;base64,AAAAIGZ0eXBtcDQy",
            "extra_args": {"max_new_tokens": 8, "_task": "any-to-any"},
        },
        options={},
    )
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("task") == "any-to-any"
    result = data.get("result") or {}
    task_output = result.get("task_output") or {}
    # Expect a modalities summary and possibly passthroughs
    mods = task_output.get("modalities")
    assert isinstance(mods, list) and len(mods) >= 1
    assert "text" in mods
    assert "image" in mods
    assert "audio" in mods
    assert "video" in mods
    # Ensure metadata present
    metadata = result.get("metadata") or {}
    assert metadata.get("task") == "any-to-any"
