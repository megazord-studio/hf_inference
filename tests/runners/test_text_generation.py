import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "google/gemma-2-2b-it",
            {"prompt": "Write a funny poem about coding in Switzerland."},
        ),
        (
            "tiiuae/falcon-rw-1b",
            {"prompt": "Write a funny poem about coding in Switzerland."},
        ),
        (
            "gpt2",
            {"prompt": "Write a funny poem about coding in Switzerland."},
        ),
        (
            "meta-llama/Llama-3.2-1B",
            {"prompt": "Write a funny poem about coding in Switzerland."},
        ),
        (
            "mistralai/Mistral-7B-v0.1",
            {"prompt": "Write a funny poem about coding in Switzerland."},
        ),
    ],
)
def test_text_generation(client, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="text-generation", payload=payload
    )
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
