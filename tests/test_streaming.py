import re
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

UUID_RE = re.compile(r"id: [0-9a-fA-F-]{36}")


# Phase 1: basic SSE streaming test for text-generation
# Keeps generation very short to minimize test latency.

def test_streaming_sse_basic():
    resp = client.get(
        "/api/inference/stream",
        params={"model_id": "gpt2", "prompt": "Hello", "max_new_tokens": 5},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: start" in body, "Missing start event"
    assert "event: token" in body, f"No token events found in SSE body: {body[:200]}..."
    assert "event: done" in body, f"No done event found in SSE body: {body[:200]}..."
    assert UUID_RE.search(body), "Correlation id (UUID) not found"
    token_events = [blk for blk in body.split("\n\n") if "event: token" in blk]
    assert token_events, "Should have at least one token event block"
    for ev in token_events:
        assert re.search(r"data: {.*}", ev), f"Token event missing JSON data: {ev}"  # noqa: W605


def test_streaming_sse_metrics_present():
    resp = client.get(
        "/api/inference/stream",
        params={"model_id": "gpt2", "prompt": "Hello world", "max_new_tokens": 6},
    )
    assert resp.status_code == 200
    body = resp.text
    done_blocks = [blk for blk in body.split("\n\n") if "event: done" in blk]
    assert done_blocks, "Missing done event block"
    done_block = done_blocks[-1]
    assert "first_token_latency_ms" in done_block, "Latency metric missing"
    assert "tokens_per_second" in done_block, "TPS metric missing"
    assert UUID_RE.search(done_block), "Done block missing correlation id"
