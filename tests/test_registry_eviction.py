import time

from app.core.registry import ModelRegistry, ModelEntry


class DummyRunner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


def test_eviction_prefers_erroring_models(monkeypatch):
    """Regression: erroring models should be evicted before healthy ones."""
    reg = ModelRegistry()

    base_time = time.time()
    monkeypatch.setattr("time.time", lambda: base_time)

    reg._models[("ok1", "text-generation")] = ModelEntry(
        model_id="ok1",
        task="text-generation",
        runner=DummyRunner("ok1"),
        status="ready",
    )
    reg._models[("err", "text-generation")] = ModelEntry(
        model_id="err",
        task="text-generation",
        runner=DummyRunner("err"),
        status="error",
    )
    reg._models[("ok2", "text-generation")] = ModelEntry(
        model_id="ok2",
        task="text-generation",
        runner=DummyRunner("ok2"),
        status="ready",
    )
    reg._models[("err", "text-generation")].last_error = "boom"

    reg._evict_one()

    assert ("err", "text-generation") not in reg._models
    assert ("ok1", "text-generation") in reg._models
    assert ("ok2", "text-generation") in reg._models


def test_eviction_uses_age_and_size(monkeypatch):
    """Regression: eviction scoring should depend on age and size.

    Exact winner depends on the chosen weights; this test encodes the current
    behavior so future changes must be explicit.
    """
    reg = ModelRegistry()

    base_time = time.time()
    monkeypatch.setattr("time.time", lambda: base_time)

    e_new = ModelEntry(
        model_id="new-small",
        task="text-generation",
        runner=DummyRunner("new-small"),
        status="ready",
    )
    e_new.last_used_at = base_time
    e_new.mem_estimate_mb = 100.0

    e_old_small = ModelEntry(
        model_id="old-small",
        task="text-generation",
        runner=DummyRunner("old-small"),
        status="ready",
    )
    e_old_small.last_used_at = base_time - 3600  # 1 hour ago
    e_old_small.mem_estimate_mb = 100.0

    e_mid_large = ModelEntry(
        model_id="mid-large",
        task="text-generation",
        runner=DummyRunner("mid-large"),
        status="ready",
    )
    e_mid_large.last_used_at = base_time - 1800  # 30 min ago
    e_mid_large.mem_estimate_mb = 2000.0

    reg._models[("new-small", "text-generation")] = e_new
    reg._models[("old-small", "text-generation")] = e_old_small
    reg._models[("mid-large", "text-generation")] = e_mid_large

    reg._evict_one()

    # With current weights (age dominates size), the oldest small model is evicted
    assert ("old-small", "text-generation") not in reg._models
    assert ("new-small", "text-generation") in reg._models
    assert ("mid-large", "text-generation") in reg._models
