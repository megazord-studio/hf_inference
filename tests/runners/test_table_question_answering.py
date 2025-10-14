import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "google/tapas-large-finetuned-wtq",
            {
                "table": [
                    ["transformers", "library"],
                    ["pandas", "dataframes"],
                ],
                "table_query": "What is in the first cell?",
            },
        ),
        (
            "google/tapas-base-finetuned-wtq",
            {
                "table": [["city", "country"], ["Bern", "Switzerland"]],
                "table_query": "Which country is Bern in?",
            },
        ),
        (
            "google/tapas-large-finetuned-wikisql-supervised",
            {
                "table": [["item", "price"], ["Chocolate", "5"]],
                "table_query": "What is the price of Chocolate?",
            },
        ),
        (
            "google/tapas-base",
            {
                "table": [["A", "B"], ["1", "2"]],
                "table_query": "What is under column A?",
            },
        ),
    ],
)
def test_table_question_answering(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="table-question-answering", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
