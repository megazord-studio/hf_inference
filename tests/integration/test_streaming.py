import re

import pytest

UUID_RE = re.compile(r"id: [0-9a-fA-F-]{36}")

PARAMS = [
    (
        "/api/inference/stream",
        {"model_id": "gpt2", "prompt": "Hello", "max_new_tokens": 5},
        ["event: start", "event: token", "event: done"],
    ),
    (
        "/api/inference/stream/text-to-image",
        {
            "model_id": "ehristoforu/stable-diffusion-v1-5-tiny",
            "prompt": "a tiny cat",
            "num_inference_steps": 4,
        },
        ["event: start", "event: progress", "event: done"],
    ),
]


@pytest.mark.parametrize("path,params,expect_events", PARAMS)
def test_streaming_endpoints_parametrized(client, path, params, expect_events):
    resp = client.get(path, params=params)
    assert resp.status_code == 200, resp.text
    body = resp.text
    for ev in expect_events:
        assert ev in body, f"Missing {ev} in SSE body"
    assert UUID_RE.search(body), "Correlation id (UUID) not found"

    # Basic sanity on progress tokens if present
    if "event: token" in body:
        token_events = [
            blk for blk in body.split("\n\n") if "event: token" in blk
        ]
        assert token_events, "Should have token events"
        for ev in token_events:
            assert "data:" in ev

    if "event: progress" in body:
        progress_blocks = [
            blk for blk in body.split("\n\n") if "event: progress" in blk
        ]
        assert progress_blocks, "Should have progress events"
        for blk in progress_blocks:
            assert "data:" in blk
