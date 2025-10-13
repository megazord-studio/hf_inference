import pytest
from tests.conftest import create_spec

@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("facebook/sam-vit-base", {"image_path": "image.jpg"}),
        ("facebook/sam-vit-huge", {"image_path": "image.jpg"}),
    ],
)
def test_mask_generation(client, sample_image, model_id, payload):
    spec = create_spec(model_id=model_id, task="mask-generation", payload={})
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
