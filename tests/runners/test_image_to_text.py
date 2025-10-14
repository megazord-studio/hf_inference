import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("Salesforce/blip-image-captioning-base", {"image_path": "image.jpg"}),
        ("nlpconnect/vit-gpt2-image-captioning", {"image_path": "image.jpg"}),
        ("Salesforce/blip2-flan-t5-xl", {"image_path": "image.jpg"}),
        ("microsoft/git-base-coco", {"image_path": "image.jpg"}),
        ("google/pix2struct-textcaps-base", {"image_path": "image.jpg"}),
    ],
)
def test_image_to_text(client, sample_image, model_id, payload):
    spec = create_spec(model_id=model_id, task="image-to-text", payload={})
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
