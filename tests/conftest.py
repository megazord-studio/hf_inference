import warnings
import pytest
from fastapi.testclient import TestClient
import os

from app.main import app

# Silence noisy third-party deprecation warnings that we cannot fix locally.
# Keep the filters narrow so that in-repo DeprecationWarnings are still visible.

# transformers OwlVitProcessor deprecation (used internally by pipelines)
warnings.filterwarnings(
    "ignore",
    message="`post_process_object_detection` method is deprecated for OwlVitProcessor and will be removed in v5.",
    category=FutureWarning,
    module=r"transformers\\.models\\.owlv2\\.processing_owlv2",
)

# SWIG-related deprecations coming from llava / swig bindings
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


@pytest.fixture(autouse=True)
def ensure_token_env(monkeypatch):
    # Provide a dummy token if not set (HF hub works w/out but keep explicit)
    monkeypatch.setenv("HF_TOKEN", os.getenv("HF_TOKEN", ""))


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def infer():
    def _infer(model_id: str, task: str, input_type: str, inputs: dict, options: dict | None = None):
        payload = {
            "model_id": model_id,
            "intent_id": None,
            "input_type": input_type,
            "inputs": inputs,
            "task": task,
            "options": options or {},
        }
        return payload
    return _infer
