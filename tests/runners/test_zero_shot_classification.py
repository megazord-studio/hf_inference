import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "facebook/bart-large-mnli",
            {
                "prompt": "This restaurant was surprisingly good for the price.",
                "candidate_labels": ["positive", "negative", "neutral"],
            },
        ),
        (
            "MoritzLaurer/DeBERTa-v3-base-mnli",
            {
                "prompt": "This restaurant was surprisingly good for the price.",
                "candidate_labels": ["positive", "negative", "neutral"],
            },
        ),
        (
            "valhalla/distilbart-mnli-12-1",
            {
                "prompt": "The new policy might increase taxes.",
                "candidate_labels": ["economy", "sports", "politics"],
            },
        ),
        (
            "roberta-large-mnli",
            {
                "prompt": "I love hiking on weekends.",
                "candidate_labels": ["travel", "hobby", "finance"],
            },
        ),
    ],
)
def test_zero_shot_classification(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="zero-shot-classification", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
