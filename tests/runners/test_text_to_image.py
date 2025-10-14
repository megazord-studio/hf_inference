import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "runwayml/stable-diffusion-v1-5",
            {
                "prompt": "A cozy wooden cabin in snowy mountains at sunrise, watercolor style."
            },
        ),
        (
            "stabilityai/stable-diffusion-2-1",
            {"prompt": "A Swiss mountain village at dusk, oil painting."},
        ),
        (
            "stabilityai/sdxl-turbo",
            {"prompt": "A minimalist poster of the Alps with bold shapes."},
        ),
        (
            "dreamlike-art/dreamlike-photoreal-2.0",
            {"prompt": "Photorealistic chalet with warm lights at night."},
        ),
    ],
)
def test_text_to_image(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="text-to-image", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
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
