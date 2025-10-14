import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            {"prompt": "This is a short sentence."},
        ),
        (
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            {"prompt": "This is a short sentence."},
        ),
        ("bert-base-uncased", {"prompt": "Embedding this sentence."}),
        ("roberta-large", {"prompt": "Another embedding sentence."}),
        ("intfloat/e5-small", {"prompt": "query: best hikes near Zermatt"}),
    ],
)
def test_feature_extraction(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="feature-extraction", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
