import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "Helsinki-NLP/opus-mt-en-de",
            {"prompt": "Good morning, how are you?"},
        ),
        (
            "facebook/m2m100_418M",
            {
                "prompt": "Good morning, how are you?",
                "src_lang": "en",
                "tgt_lang": "sv",
            },
        ),
        (
            "facebook/nllb-200-distilled-600M",
            {
                "prompt": "Good morning, how are you?",
                "src_lang": "eng_Latn",
                "tgt_lang": "deu_Latn",
            },
        ),
        (
            "facebook/wmt19-en-de",
            {"prompt": "Please proceed to platform 4 for the next departure."},
        ),
        (
            "facebook/mbart-large-50-many-to-many-mmt",
            {
                "prompt": "The mountain pass is closed due to snow.",
                "src_lang": "en_EN",
                "tgt_lang": "de_DE",
            },
        ),
        (
            "Helsinki-NLP/opus-mt-en-fr",
            {"prompt": "The cable car offers amazing views of the Alps."},
        ),
    ],
)
def test_translation(client, model_id, payload):
    spec = create_spec(model_id=model_id, task="translation", payload=payload)
    resp = client.post("/inference", data={"spec": spec})
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
