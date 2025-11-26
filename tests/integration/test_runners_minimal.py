import pytest

PARAMS = [
    ("google/vit-base-patch16-224", 'image-classification', 'image', {'image_base64': None}, 'predictions'),
    ("nlpconnect/vit-gpt2-image-captioning", 'image-captioning', 'image', {'image_base64': None}, 'text'),
    ("hustvl/yolos-tiny", 'object-detection', 'image', {'image_base64': None}, 'detections'),
    ("nvidia/segformer-b0-finetuned-ade-512-512", 'image-segmentation', 'image', {'image_base64': None}, 'labels'),
]


def _mk_image_b64(color=(120,34,56), size=(32,32)):
    from PIL import Image
    import io, base64
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


@pytest.mark.parametrize("model_id,task,input_type,inputs,expected", PARAMS)
def test_runners_minimal_parametrized(client, model_id, task, input_type, inputs, expected):
    if input_type == 'image' and inputs.get('image_base64') is None:
        inputs['image_base64'] = _mk_image_b64()

    resp = client.post('/api/inference', json={
        'model_id': model_id,
        'intent_id': '',
        'input_type': input_type,
        'inputs': inputs,
        'task': task,
        'options': {},
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()['result']
    assert data.get('task') == task
    out = data['task_output']
    assert expected in out or any(k == expected for k in out.keys())

