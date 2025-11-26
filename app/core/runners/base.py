"""Base runner abstraction (Phase 0)."""
from __future__ import annotations
from typing import Dict, Any

class BaseRunner:
    def __init__(self, model_id: str, device):
        self.model_id = model_id
        self.device = device
        self._loaded = False

    def load(self) -> int:
        """Load model weights. Return param count (approx)."""
        raise NotImplementedError

    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        raise NotImplementedError

    def unload(self) -> None:
        # Allow GC by dropping references; DRY & minimal side effects
        pass

