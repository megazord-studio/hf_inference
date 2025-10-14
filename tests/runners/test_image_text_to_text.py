import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


def _clean(p):
    return {k: v for k, v in p.items() if not k.endswith("_path")}


@pytest.mark.parametrize(
    "model_id,payload",
    [
        (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            {
                "image_path": "image.jpg",
                "prompt": "Give a concise caption and mention one color.",
            },
        ),
        (
            "HuggingFaceM4/idefics2-8b",
            {
                "image_path": "image.jpg",
                "prompt": "Give a concise caption and mention one color.",
            },
        ),
        (
            "llava-hf/llava-1.5-7b-hf",
            {
                "image_path": "image.jpg",
                "prompt": "Describe this scene in one sentence.",
            },
        ),
        (
            "openbmb/MiniCPM-Llama3-V-2_5",
            {
                "image_path": "image.jpg",
                "prompt": "Give a concise caption and mention one color.",
            },
        ),
        (
            "01-ai/Yi-VL-6B",
            {
                "image_path": "image.jpg",
                "prompt": "Give a concise caption and mention one color.",
            },
        ),
        (
            "OpenGVLab/InternVL2-8B",
            {
                "image_path": "image.jpg",
                "prompt": "Give a concise caption and mention one color.",
            },
        ),
        (
            "Salesforce/blip2-opt-2.7b",
            {
                "image_path": "image.jpg",
                "prompt": "Caption this photo with one color mentioned.",
            },
        ),
        (
            "microsoft/kosmos-2-patch14-224",
            {
                "image_path": "image.jpg",
                "prompt": "Brief caption, include one color.",
            },
        ),
        (
            "microsoft/Florence-2-base-ft",
            {
                "image_path": "image.jpg",
                "prompt": "Brief caption, include one color.",
            },
        ),
        (
            "google/paligemma-3b-pt-224",
            {
                "image_path": "image.jpg",
                "prompt": "One-sentence caption with a color.",
            },
        ),
        (
            "THUDM/cogvlm2-llama3-chat-19B",
            {
                "image_path": "image.jpg",
                "prompt": "Brief caption mentioning an object.",
            },
        ),
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            {"image_path": "image.jpg", "prompt": "Describe in one sentence."},
        ),
    ],
)
def test_image_text_to_text(client, sample_image, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="image-text-to-text", payload=_clean(payload)
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
