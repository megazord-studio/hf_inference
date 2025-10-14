import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("facebook/detr-resnet-50-panoptic", {"image_path": "image.jpg"}),
        (
            "facebook/mask2former-swin-base-coco-panoptic",
            {"image_path": "image.jpg"},
        ),
        (
            "nvidia/segformer-b0-finetuned-ade-512-512",
            {"image_path": "image.jpg"},
        ),
        (
            "facebook/mask2former-swin-large-coco-panoptic",
            {"image_path": "image.jpg"},
        ),
        (
            "nvidia/segformer-b3-finetuned-ade-512-512",
            {"image_path": "image.jpg"},
        ),
    ],
)
def test_image_segmentation(client, sample_image, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="image-segmentation", payload={}
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
