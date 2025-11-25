"""ModelRegistry - minimal Phase 0 implementation.

Responsibilities:
- Lazily load text models (generation, classification, embeddings).
- Provide predict(task, model_id, inputs, options) interface.
- Simple LRU eviction based on max_loaded_models env.
- Memory estimation (param_count * dtype_bytes) best-effort.

DRY & KISS: keep logic straightforward; no side effects outside in-memory state.
"""
from __future__ import annotations
import time
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple

from app.core.device import select_device
from app.core.runners import (
    get_runner_cls,
    SUPPORTED_TASKS,
)
from app.core.resources import RESOURCES

_LOCK = threading.RLock()


@dataclass
class ModelEntry:
    model_id: str
    task: str
    runner: Any
    status: str  # loading|ready|error
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    param_count: Optional[int] = None
    mem_estimate_mb: Optional[float] = None
    popularity_score: Optional[float] = None  # placeholder (Phase 0 not computed)
    acceleration_profile: Optional[dict] = None  # placeholder for Phase 3

    def touch(self) -> None:
        self.last_used_at = time.time()


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[Tuple[str, str], ModelEntry] = {}
        self._device = select_device(os.getenv("FORCE_DEVICE", "auto"))
        self._max_loaded = int(os.getenv("MAX_LOADED_MODELS", "4"))

    # --- public API ---
    def predict(self, task: str, model_id: str, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        entry = self._get_or_load(task, model_id)
        entry.touch()
        start = time.time()
        output = entry.runner.predict(inputs, options)
        runtime_ms = int((time.time() - start) * 1000)
        resolved_id = getattr(entry.runner, 'resolved_model_id', None)
        return {"task": task, "model_id": model_id, "output": output, "runtime_ms": runtime_ms, "backend": getattr(entry.runner, 'backend', 'torch'), "resolved_model_id": resolved_id}

    def list_loaded(self) -> Dict[str, Any]:  # debug/inspection
        with _LOCK:
            return {
                f"{mid}:{task}": {
                    "status": e.status,
                    "last_used": int(time.time() - e.last_used_at),
                    "mem_mb": e.mem_estimate_mb,
                } for (mid, task), e in self._models.items()
            }

    def unload(self, task: str, model_id: str) -> bool:
        with _LOCK:
            # Remove entries matching the exact model_id or variant ending with '/model_id'
            keys_to_remove = [
                key for key in self._models.keys()
                if key[0] == model_id or key[0].endswith(f"/{model_id}")
            ]
            if not keys_to_remove:
                return False
            for key in keys_to_remove:
                try:
                    self._models[key].runner.unload()
                except Exception:
                    pass
                del self._models[key]
            return True

    # --- internals ---
    def _get_or_load(self, task: str, model_id: str) -> ModelEntry:
        key = (model_id, task)
        with _LOCK:
            existing = self._models.get(key)
            if existing:
                return existing
            if len(self._models) >= self._max_loaded or RESOURCES.need_eviction():
                self._evict_one()
            runner_cls = get_runner_cls(task)
            runner = runner_cls(model_id=model_id, device=self._device)
            entry = ModelEntry(model_id=model_id, task=task, runner=runner, status="loading")
            try:
                param_count = runner.load()
                entry.status = "ready"
                entry.param_count = param_count
                if param_count:
                    # assume 2 bytes per param if half precision possible, else 4 bytes
                    bytes_per_param = 2 if getattr(runner, "dtype", None) == "float16" else 4
                    entry.mem_estimate_mb = round((param_count * bytes_per_param) / (1024**2), 2)
            except Exception as e:
                msg = str(e)
                if task == 'text-generation' and 'does not appear to have a file named' in msg:
                    try:
                        from app.core.runners.text_onnx import OnnxTextGenerationRunner
                        runner = OnnxTextGenerationRunner(model_id=model_id, device=self._device)
                        entry.runner = runner
                        param_count = runner.load()
                        entry.status = 'ready'
                        entry.param_count = param_count or None
                        entry.mem_estimate_mb = None
                    except Exception as onnx_e:
                        entry.status = 'error'
                        entry.runner = runner
                        raise RuntimeError(f"Failed ONNX fallback for {model_id}: {onnx_e}") from onnx_e
                else:
                    # No dummy fallback for vision/audio; surface error directly
                    entry.status = "error"
                    entry.runner = runner
                    raise RuntimeError(f"Failed loading model {model_id} for task {task}: {e}")
            self._models[key] = entry
            return entry

    def _evict_one(self) -> None:
        # Choose least recently used ready model
        lru_key = None
        lru_ts = time.time()
        for key, entry in self._models.items():
            if entry.status != "ready":
                continue
            if entry.last_used_at < lru_ts:
                lru_ts = entry.last_used_at
                lru_key = key
        if lru_key:
            try:
                self._models[lru_key].runner.unload()
            except Exception:
                pass
            del self._models[lru_key]

# Singleton instance
REGISTRY = ModelRegistry()

# Mapping pipeline_tag -> default task for Phase 0 + Phase 2
PIPELINE_TO_TASK = {
    "text-generation": "text-generation",
    "text-classification": "text-classification",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",  # map to embedding for now
    # Phase 2 mappings
    "image-to-text": "image-captioning",
    "image-classification": "image-classification",
    "object-detection": "object-detection",
    "image-segmentation": "image-segmentation",
    "depth-estimation": "depth-estimation",
    "automatic-speech-recognition": "automatic-speech-recognition",
    "audio-classification": "audio-classification",
    "text-to-speech": "text-to-speech",
    # Phase A new pipeline tags (stubbed runners pending)
    "zero-shot-image-classification": "zero-shot-image-classification",
    "zero-shot-object-detection": "zero-shot-object-detection",
    "keypoint-detection": "keypoint-detection",
    "image-super-resolution": "image-super-resolution",
    "image-restoration": "image-restoration",
    "image-to-3d": "image-to-3d",
    "text-to-3d": "text-to-3d",
    "text-to-image": "text-to-image",
    "image-to-image": "image-to-image",
    "text-to-video": "text-to-video",
    "image-to-video": "image-to-video",
    "audio-to-audio": "audio-to-audio",
    "text-to-audio": "text-to-audio",
    "audio-text-to-text": "audio-text-to-text",
    "voice-activity-detection": "voice-activity-detection",
    "time-series-forecasting": "time-series-forecasting",
    "visual-document-retrieval": "visual-document-retrieval",
    "any-to-any": "any-to-any",
    "image_text-to-text": "image-text-to-text",
}

__all__ = ["REGISTRY", "PIPELINE_TO_TASK", "ModelRegistry"]
