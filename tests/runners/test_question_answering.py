import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "deepset/roberta-base-squad2",
            {
                "qa_question": "Who wrote Faust?",
                "qa_context": "Johann Wolfgang von Goethe was a German writer and statesman.",
            },
        ),
        (
            "distilbert-base-uncased-distilled-squad",
            {
                "qa_question": "Who wrote Faust?",
                "qa_context": "Johann Wolfgang von Goethe was a German writer and statesman.",
            },
        ),
        (
            "deepset/bert-base-cased-squad2",
            {
                "qa_question": "What is the capital of Switzerland?",
                "qa_context": "Bern is the de facto capital of Switzerland.",
            },
        ),
        (
            "deepset/xlm-roberta-large-squad2",
            {
                "qa_question": "What does SBB refer to?",
                "qa_context": "SBB is the national railway company of Switzerland.",
            },
        ),
        (
            "deepset/minilm-uncased-squad2",
            {
                "qa_question": "What language is mainly spoken in Zurich?",
                "qa_context": "In Zurich, Swiss German is predominantly spoken.",
            },
        ),
    ],
)
def test_question_answering(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="question-answering", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
