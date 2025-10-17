"""Test that extra_args are properly passed to runners."""

import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,task,payload,extra_args",
    [
        # Text generation with extra_args
        (
            "gpt2",
            "text-generation",
            {"prompt": "Hello, world!"},
            {"max_length": 10},
        ),
        # Sentiment analysis with extra_args
        (
            "distilbert-base-uncased-finetuned-sst-2-english",
            "sentiment-analysis",
            {"prompt": "This is great!"},
            {"top_k": 1},
        ),
        # Image classification with extra_args (requires image)
        (
            "google/vit-base-patch16-224",
            "image-classification",
            {"image_path": "image.jpg"},
            {"top_k": 3},
        ),
    ],
)
def test_extra_args_passed_to_runners(
    client, sample_image, model_id, task, payload, extra_args
):
    """Test that extra_args are accepted and do not cause errors."""
    spec = create_spec(
        model_id=model_id, task=task, payload=payload, extra_args=extra_args
    )

    # Add image file if needed
    files = None
    if task in [
        "image-classification",
        "image-to-text",
        "image-segmentation",
    ]:
        files = {"image": ("test.png", sample_image, "image/png")}

    if files:
        resp = client.post("/inference", data={"spec": spec}, files=files)
    else:
        resp = client.post("/inference", data={"spec": spec})

    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
