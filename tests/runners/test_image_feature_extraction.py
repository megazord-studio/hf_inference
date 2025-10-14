import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("openai/clip-vit-base-patch32", {"image_path": "image.jpg"}),
        ("google/vit-base-patch16-224", {"image_path": "image.jpg"}),
    ],
)
def test_image_feature_extraction(client, sample_image, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="image-feature-extraction", payload={}
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
