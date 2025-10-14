import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "roberta-base",
            {
                "mask_sentence": "The capital of Switzerland is <mask>.",
                "mask_sentence_alt": "The capital of Switzerland is [MASK].",
            },
        ),
        (
            "bert-base-uncased",
            {
                "mask_sentence": "the capital of switzerland is [MASK].",
                "mask_sentence_alt": "the capital of switzerland is <mask>.",
            },
        ),
        (
            "albert-base-v2",
            {
                "mask_sentence": "The famous mountain in Switzerland is [MASK].",
                "mask_sentence_alt": "The famous mountain in Switzerland is <mask>.",
            },
        ),
        (
            "distilroberta-base",
            {
                "mask_sentence": "Swiss chocolate is <mask>.",
                "mask_sentence_alt": "Swiss chocolate is [MASK].",
            },
        ),
        (
            "google/electra-base-generator",
            {
                "mask_sentence": "The city of Basel is known for its [MASK] fair.",
                "mask_sentence_alt": "The city of Basel is known for its <mask> fair.",
            },
        ),
    ],
)
def test_fill_mask(client, model_id, payload):
    spec = create_spec(model_id=model_id, task="fill-mask", payload=payload)
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
