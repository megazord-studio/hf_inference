import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


def _clean(p):  # drop *_path hints
    return {k: v for k, v in p.items() if not k.endswith("_path")}


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "dandelin/vilt-b32-finetuned-vqa",
            {"image_path": "image.jpg", "question": "What is on the table?"},
        ),
        (
            "Salesforce/blip-vqa-base",
            {
                "image_path": "image.jpg",
                "question": "What color are the chairs?",
            },
        ),
        (
            "microsoft/florence-2-base-ft",
            {
                "image_path": "image.jpg",
                "question": "Summarize the image in one sentence.",
            },
        ),
    ],
)
def test_visual_question_answering(client, sample_image, model_id, payload):
    spec = create_spec(
        model_id=model_id,
        task="visual-question-answering",
        payload=_clean(payload),
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
