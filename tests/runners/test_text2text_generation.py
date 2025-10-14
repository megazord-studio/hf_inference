import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "google/flan-t5-base",
            {
                "prompt": "Translate to German: The hotel staff are very friendly and helpful."
            },
        ),
        (
            "facebook/bart-base",
            {
                "prompt": "Translate to German: The hotel staff are very friendly and helpful."
            },
        ),
        (
            "google/mt5-small",
            {
                "prompt": "Translate to German: The hotel staff are very friendly and helpful."
            },
        ),
        (
            "t5-small",
            {
                "prompt": "Summarize: Switzerland has scenic trains connecting the Alps."
            },
        ),
        (
            "bigscience/T0pp",
            {
                "prompt": "Paraphrase: The software was quick to install and easy to use."
            },
        ),
    ],
)
def test_text2text_generation(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="text2text-generation", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
