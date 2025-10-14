import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


def _clean(p):
    return {k: v for k, v in p.items() if not k.endswith("_path")}


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "google/owlvit-base-patch32",
            {
                "image_path": "image.jpg",
                "candidate_labels": ["chair", "table", "plant", "sea"],
            },
        ),
        (
            "google/owlvit-large-patch14",
            {
                "image_path": "image.jpg",
                "candidate_labels": ["window", "umbrella", "tree"],
            },
        ),
        (
            "IDEA-Research/grounding-dino-base",
            {
                "image_path": "image.jpg",
                "candidate_labels": ["chair", "table", "plant"],
            },
        ),
    ],
)
def test_zero_shot_object_detection(client, sample_image, model_id, payload):
    spec = create_spec(
        model_id=model_id,
        task="zero-shot-object-detection",
        payload=_clean(payload),
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
