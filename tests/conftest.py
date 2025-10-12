"""
Pytest fixtures for FastAPI inference tests.
"""

import pytest
import json
import io
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (100, 100), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


@pytest.fixture
def sample_audio():
    """Create a sample audio file (placeholder)."""
    # For testing purposes, use a minimal WAV file
    # This is a valid but silent 1-second WAV file at 16kHz
    import wave
    import numpy as np
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        # 1 second of silence
        wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    buf.seek(0)
    return buf


@pytest.fixture
def sample_video():
    """Create a placeholder for video file."""
    # For testing purposes, return a minimal placeholder
    # Real video testing would require actual video files or generation
    buf = io.BytesIO(b"placeholder video content")
    buf.seek(0)
    return buf


def create_spec(model_id: str, task: str, payload: dict = None) -> str:
    """Helper to create spec JSON string."""
    return json.dumps({
        "model_id": model_id,
        "task": task,
        "payload": payload or {}
    })
