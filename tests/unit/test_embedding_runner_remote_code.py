from __future__ import annotations

import pytest

from app.core.runners.text import EmbeddingRunner


class _SpyST:
    def __init__(self, model_id, **kwargs):
        # record inputs for assertions
        _SpyST.called_with = (model_id, kwargs)
        # minimal interface mimicking SentenceTransformer
        self._modules = {}

    def encode(self, texts):  # not used in this test
        return [[0.0]]


@pytest.fixture(autouse=True)
def patch_sentence_transformers(monkeypatch):
    # Patch in the class in the module where it's used
    from app.core.runners import text as text_mod

    monkeypatch.setattr(text_mod, "SentenceTransformer", _SpyST, raising=True)
    yield


def test_embedding_runner_uses_trust_remote_code_true(monkeypatch):
    r = EmbeddingRunner(model_id="jinaai/jina-embeddings-v3", device=None)
    params = r.load()
    assert isinstance(params, int)
    # Ensure our spy captured call
    assert hasattr(_SpyST, "called_with")
    model_id, kwargs = _SpyST.called_with
    assert model_id == "jinaai/jina-embeddings-v3"
    # The loader should pass trust_remote_code=True when supported
    # Our spy accepts arbitrary kwargs so presence is what we check
    assert kwargs.get("trust_remote_code") is True
