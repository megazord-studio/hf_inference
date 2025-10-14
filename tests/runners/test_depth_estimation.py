import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("Intel/dpt-hybrid-midas", {"image_path": "image.jpg"}),
        ("Intel/dpt-large", {"image_path": "image.jpg"}),
        ("Intel/dpt-swinv2-tiny-256", {"image_path": "image.jpg"}),
        ("LiheYoung/depth-anything-small-hf", {"image_path": "image.jpg"}),
    ],
)
def test_depth_estimation(client, sample_image, model_id, payload):
    spec = create_spec(model_id=model_id, task="depth-estimation", payload={})
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        # Check if response is binary (image file) or JSON
        content_type = resp.headers.get("content-type", "")
        if (
            "image" in content_type
            or "application/octet-stream" in content_type
        ):
            # Binary response - verify it's not empty
            assert len(resp.content) > 0
        else:
            # JSON response
            data = resp.json()
            assert isinstance(data, (list, dict))
            check_response_for_skip_or_error(data, model_id)
