import pytest
from tests.conftest import create_spec, check_response_for_skip_or_error

def _clean(p):
    return {k: v for k, v in p.items() if not k.endswith("_path")}

@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("impira/layoutlm-document-qa", {"image_path": "image.jpg", "question": "What is the total amount?"}),
        ("naver-clova-ix/donut-base-finetuned-docvqa", {"image_path": "image.jpg", "question": "What is the total amount?"}),
        ("microsoft/layoutlmv3-base", {"image_path": "image.jpg", "question": "Who is the recipient?"}),
        ("navervision/lectrobase", {"image_path": "image.jpg", "question": "What is the invoice number?"}),
        ("impira/layoutlmv2-finetuned-docvqa", {"image_path": "image.jpg", "question": "What is the due date?"}),
    ],
)
def test_document_question_answering(client, sample_image, model_id, payload):
    spec = create_spec(model_id=model_id, task="document-question-answering", payload=_clean(payload))
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
