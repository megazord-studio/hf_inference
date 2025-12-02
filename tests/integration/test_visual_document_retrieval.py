import pytest


def _make_image_b64(color=(200, 120, 30), size=(32, 32)):
    import base64
    import io
    from PIL import Image

    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


@pytest.mark.parametrize(
    "model_id",
    [
        "vidore/colqwen2.5-v0.2",  # VLM via trust_remote_code
        "jinaai/jina-embeddings-v4",  # Jina model (trust_remote_code)
    ],
)
def test_visual_document_retrieval_models(client, model_id):
    img_b64 = _make_image_b64()
    resp = client.post(
        "/api/inference",
        json={
            "model_id": model_id,
            "intent_id": "",
            "input_type": "image",
            "inputs": {"image_base64": img_b64},
            "task": "visual-document-retrieval",
            "options": {"k": 3},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]["task_output"]
    assert "results" in data and isinstance(data["results"], list)
    assert len(data["results"]) >= 1
    assert all("doc_id" in r and "score" in r for r in data["results"]) 
