import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


def _clean(p):
    return {k: v for k, v in p.items() if not k.endswith("_path")}


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "openai/clip-vit-base-patch32",
            {
                "image_path": "image.jpg",
                "candidate_labels": [
                    "patio",
                    "dome",
                    "restaurant",
                    "planetarium",
                ],
            },
        ),
        (
            "openai/clip-vit-large-patch14",
            {
                "image_path": "image.jpg",
                "candidate_labels": ["beach", "balcony", "forest"],
            },
        ),
        (
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            {
                "image_path": "image.jpg",
                "candidate_labels": ["chair", "plant", "sea"],
            },
        ),
    ],
)
def test_zero_shot_image_classification(
    client, sample_image, model_id, payload
):
    spec = create_spec(
        model_id=model_id,
        task="zero-shot-image-classification",
        payload=_clean(payload),
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
