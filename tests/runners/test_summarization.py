import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "facebook/bart-large-cnn",
            {"prompt": "Switzerland trains are punctual and efficient."},
        ),
        (
            "sshleifer/distilbart-cnn-12-6",
            {"prompt": "Switzerland trains are punctual and efficient."},
        ),
        (
            "google/pegasus-cnn_dailymail",
            {
                "prompt": "Swiss cities consistently rank high for quality of life due to safety and infrastructure."
            },
        ),
        (
            "t5-base",
            {
                "prompt": "The Swiss rail network is known for seamless connections and scenic routes."
            },
        ),
        (
            "philschmid/bart-large-cnn-samsum",
            {
                "prompt": "A: Are you coming to Zurich? B: Yes, I will arrive at 3 PM by train."
            },
        ),
    ],
)
def test_summarization(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="summarization", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
