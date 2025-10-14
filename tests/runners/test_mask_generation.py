from tests.conftest import create_spec


def test_mask_generation(client, sample_image):
    spec = create_spec(
        model_id="facebook/sam-vit-base", task="mask-generation", payload={}
    )
    files = {"image": ("test.png", sample_image, "image/png")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    expected = {
        "error": "mask-generation unsupported",
        "reason": "Segment Anything models are not exposed via transformers.pipeline.",
        "hint": "Use facebook/sam... with the segment-anything library, or switch to an image-segmentation model.",
    }
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        assert data == expected
