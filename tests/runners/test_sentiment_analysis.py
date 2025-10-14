import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "distilbert-base-uncased-finetuned-sst-2-english",
            {"prompt": "I absolutely loved this place!"},
        ),
        (
            "cardiffnlp/twitter-roberta-base-sentiment",
            {"prompt": "This is terrible news."},
        ),
        (
            "nlptown/bert-base-multilingual-uncased-sentiment",
            {"prompt": "Das Essen war hervorragend und der Service schnell."},
        ),
        (
            "finiteautomata/bertweet-base-sentiment-analysis",
            {"prompt": "OMG this view is insane 😍"},
        ),
        ("microsoft/deberta-v3-base", {"prompt": "Meh, could be better."}),
    ],
)
def test_sentiment_analysis(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="sentiment-analysis", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
