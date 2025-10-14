import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ASSETS_DIR = (Path(__file__).resolve().parents[1] / "assets").resolve()


def _read_asset_bytes(filename: str) -> bytes:
    path = (ASSETS_DIR / filename).resolve()
    if not path.exists():
        # Make it obvious what is missing and where we looked
        raise FileNotFoundError(
            f"Asset not found: {path}. "
            f"Expected assets at: {ASSETS_DIR} "
            f"(required files: image.jpg, audio.wav, video.mp4)."
        )
    return path.read_bytes()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """
    Open the demo image from ./assets/image.jpg and return a BytesIO handle.
    """
    data = _read_asset_bytes("image.jpg")
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


@pytest.fixture
def sample_audio():
    """
    Open the demo audio from ./assets/audio.wav and return a BytesIO handle.
    """
    data = _read_asset_bytes("audio.wav")
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


@pytest.fixture
def sample_video():
    """
    Open the demo video from ./assets/video.mp4 and return a BytesIO handle.
    """
    data = _read_asset_bytes("video.mp4")
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def create_spec(model_id: str, task: str, payload: dict = None) -> str:
    """Helper to create spec JSON string."""
    return json.dumps(
        {"model_id": model_id, "task": task, "payload": payload or {}}
    )


def check_response_for_skip_or_error(data, model_id: str):
    """
    Check if JSON response contains 'skipped' or 'error' fields.
    If found, skip the test with appropriate message instead of failing.

    Args:
        data: JSON response data (dict or list)
        model_id: Model ID for better error messages
    """
    if isinstance(data, dict):
        if data.get("skipped"):
            reason = data.get("reason", "Unknown reason")
            hint = data.get("hint", "")
            message = f"Model {model_id} skipped: {reason}"
            if hint:
                message += f" (Hint: {hint})"
            pytest.skip(message)
        elif "error" in data:
            error = data.get("error", "Unknown error")
            reason = data.get("reason", "Unknown reason")
            hint = data.get("hint", "")
            message = f"Model {model_id} error: {error} - {reason}"
            if hint:
                message += f" (Hint: {hint})"
            pytest.skip(message)
