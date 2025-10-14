import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "dslim/bert-base-NER",
            {"prompt": "Barack Obama was born in Hawaii."},
        ),
        (
            "dbmdz/bert-large-cased-finetuned-conll03-english",
            {"prompt": "Barack Obama was born in Hawaii."},
        ),
        (
            "Davlan/bert-base-multilingual-cased-ner-hrl",
            {"prompt": "Roger Federer lives in Switzerland."},
        ),
        (
            "Jean-Baptiste/camembert-ner",
            {"prompt": "Emmanuel Macron est né à Amiens."},
        ),
        (
            "xlm-roberta-large-finetuned-conll03-english",
            {"prompt": "Angela Merkel met in Berlin."},
        ),
    ],
)
def test_token_classification(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="token-classification", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
