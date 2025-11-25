import re
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

UUID_RE = re.compile(r"id: [0-9a-fA-F-]{36}")


def test_text_to_image_streaming_progress_events():
    """Ensure text-to-image streaming yields progress and done events.

    Uses the same diffusion model id as other vision tests for consistency.
    """
    # Model id chosen to align with existing vision generation tests; adjust if needed.
    model_id = "ehristoforu/stable-diffusion-v1-5-tiny"
    resp = client.get(
        "/api/inference/stream/text-to-image",
        params={"model_id": model_id, "prompt": "a tiny cat", "num_inference_steps": 4},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: start" in body, "Missing start event"
    assert "event: progress" in body, f"No progress events found in SSE body: {body[:200]}..."
    assert "event: done" in body, f"No done event found in SSE body: {body[:200]}..."
    assert UUID_RE.search(body), "Correlation id (UUID) not found"

    progress_blocks = [blk for blk in body.split("\n\n") if "event: progress" in blk]
    assert progress_blocks, "Should have at least one progress event block"
    for blk in progress_blocks:
        m = re.search(r"data: (\{.*\})", blk)  # noqa: W605
        assert m, f"Progress event missing JSON data: {blk}"
        payload = eval(m.group(1))  # tests use trusted backend; keep simple
        assert 1 <= payload["step"] <= payload["total_steps"]
        assert 0.0 <= payload["percent"] <= 100.0

    done_blocks = [blk for blk in body.split("\n\n") if "event: done" in blk]
    done = done_blocks[-1]
    assert "image_base64" in done, "Done event missing image_base64"


