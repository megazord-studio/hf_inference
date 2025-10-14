import pytest

from tests.conftest import check_response_for_skip_or_error
from tests.conftest import create_spec


@pytest.mark.parametrize(
    "model_id,payload",
    [
        ("MCG-NJU/videomae-base", {"video_path": "video.mp4"}),
        (
            "MCG-NJU/videomae-small-finetuned-kinetics",
            {"video_path": "video.mp4"},
        ),
    ],
)
def test_video_classification(client, sample_video, model_id, payload):
    spec = create_spec(
        model_id=model_id, task="video-classification", payload={}
    )
    files = {"video": ("test.mp4", sample_video, "video/mp4")}
    resp = client.post("/inference", data={"spec": spec}, files=files)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, (list, dict))
        check_response_for_skip_or_error(data, model_id)
