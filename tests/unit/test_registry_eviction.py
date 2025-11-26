import time
from app.core.registry import REGISTRY
from app.core import runners


class DummyRunner:
    def __init__(self, model_id: str, device=None, dtype=None):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._loaded = False
        self.unloaded = False

    def load(self) -> int:
        # Pretend param counts differ by model id for predictable mem estimates
        # e.g., "m1" -> 10M params, "m2" -> 5M, "m3" -> 3M
        base = {
            "m1": 10_000_000,
            "m2": 5_000_000,
            "m3": 3_000_000,
        }
        cnt = base.get(self.model_id, 1_000_000)
        self._loaded = True
        # Simulate small delay
        time.sleep(0.01)
        return cnt

    def predict(self, inputs, options):
        if not self._loaded:
            raise RuntimeError("not loaded")
        return {"ok": True}

    def unload(self) -> None:
        # mark unloaded for assertions
        self.unloaded = True
        self._loaded = False


def test_registry_eviction_monkeypatch(monkeypatch):
    # Monkeypatch runner registry to return DummyRunner for any task
    monkeypatch.setattr(runners, "TASK_TO_RUNNER", {"text-classification": DummyRunner})

    # Ensure a clean registry state
    REGISTRY._models.clear()
    REGISTRY._loading_futures.clear()

    # Set tiny limits so eviction will trigger
    REGISTRY._max_loaded = 2
    # Set memory limit low: m1 (10M params ~ 40MB), m2 ~20MB, m3 ~12MB. Use 30MB
    REGISTRY._memory_limit_mb = 30

    # Load m1 and m2; both should be present
    e1 = REGISTRY._sync_load_model("text-classification", "m1")
    e2 = REGISTRY._sync_load_model("text-classification", "m2")

    assert ("m1", "text-classification") in REGISTRY._models
    assert ("m2", "text-classification") in REGISTRY._models

    # Keep references to original runners so we can assert unload happened
    r1 = e1.runner
    r2 = e2.runner

    # Now load m3 which should trigger eviction to respect count and memory
    e3 = REGISTRY._sync_load_model("text-classification", "m3")

    # After loading, ensure we have at most _max_loaded models
    assert len(REGISTRY._models) <= REGISTRY._max_loaded

    # Ensure at least one previous runner was unloaded (unload sets flag)
    assert r1.unloaded or r2.unloaded

    # Also ensure total memory below limit
    assert REGISTRY._total_mem_used_mb() <= REGISTRY._memory_limit_mb

    # Clean up state for other tests
    REGISTRY._models.clear()
    REGISTRY._loading_futures.clear()
