import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


def _clean(p):
    return {k: v for k, v in p.items() if not k.endswith("_path")}


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "runwayml/stable-diffusion-inpainting",
            {
                "init_image_path": "image.jpg",
                "prompt": "Make the sky sunset orange.",
            },
        ),
        (
            "stabilityai/stable-diffusion-2-inpainting",
            {"init_image_path": "image.jpg", "prompt": "Turn the chairs red."},
        ),
    ],
)
def test_image_to_image(client, sample_image, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="image-to-image", payload=_clean(payload)
    )
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
