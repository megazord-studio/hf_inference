def test_infer_fixture_builds_payload(infer):
    payload = infer(
        model_id="Xenova/gpt2",
        task="text-generation",
        input_type="text",
        inputs={"text": "hello"},
        options={"max_new_tokens": 4},
    )
    assert payload["task"] == "text-generation"
    assert payload["inputs"]["text"] == "hello"

