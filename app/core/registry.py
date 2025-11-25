"""ModelRegistry - Phase B implementation.

Responsibilities:
- Lazily load models for all supported tasks (text, vision, audio, diffusion, video).
- Provide predict(task, model_id, inputs, options) interface.
- Async-aware loading for heavy models to avoid blocking inference requests.
- Weighted eviction (LRU + size + last error) with simple memory watermark check.
- Memory estimation (param_count * dtype_bytes) best-effort.

DRY & KISS: keep logic straightforward; no side effects outside in-memory state.
"""
from __future__ import annotations
import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple

from app.core.device import select_device, choose_dtype
from app.core.runners import (
    get_runner_cls,
    SUPPORTED_TASKS,
)
from app.core.resources import RESOURCES

_LOCK = threading.RLock()
_HEAVY_TASKS = {
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
    "image-to-3d",
    "text-to-3d",
    "any-to-any",
}


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
    load_started_at: Optional[float] = None
    load_finished_at: Optional[float] = None
    load_time_ms: Optional[int] = None
    last_error: Optional[str] = None

    def touch(self) -> None:
        self.last_used_at = time.time()


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[Tuple[str, str], ModelEntry] = {}
        self._device = select_device("auto")
        # Hardcode a small cap; tune in code instead of env.
        self._max_loaded = 4
        # Async loading infra
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._loading_futures: Dict[Tuple[str, str], asyncio.Future] = {}
        # Limit number of concurrent heavy loads
        self._max_concurrent_loads = 2
        self._load_semaphore = asyncio.Semaphore(self._max_concurrent_loads)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

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
        return {
            "task": task,
            "model_id": model_id,
            "output": output,
            "runtime_ms": runtime_ms,
            "backend": getattr(entry.runner, 'backend', 'torch'),
            "resolved_model_id": resolved_id,
        }

    def list_loaded(self) -> Dict[str, Any]:  # debug/inspection
        with _LOCK:
            now = time.time()
            return {
                f"{mid}:{task}": {
                    "status": e.status,
                    "last_used_s_ago": int(now - e.last_used_at),
                    "loaded_s_ago": int(now - e.loaded_at) if e.loaded_at else None,
                    "mem_mb": e.mem_estimate_mb,
                    "load_time_ms": e.load_time_ms,
                    "last_error": e.last_error,
                }
                for (mid, task), e in self._models.items()
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
            # Existing error entries should be retried on demand
            if len(self._models) >= self._max_loaded or RESOURCES.need_eviction():
                self._evict_one()

            if task in _HEAVY_TASKS:
                # Use async loading queue for heavy models, blocking caller only while waiting
                future = self._loading_futures.get(key)
                if future is None or future.done():
                    future = asyncio.run_coroutine_threadsafe(
                        self._async_load_model(task, model_id), self._loop
                    )
                    self._loading_futures[key] = future
            runner_entry_future = self._loading_futures.get(key)
        if task in _HEAVY_TASKS and runner_entry_future is not None:
            # Block current thread until model is ready or error raised
            return runner_entry_future.result()
        # Non-heavy models: load synchronously in current thread
        return self._sync_load_model(task, model_id)

    async def _async_load_model(self, task: str, model_id: str) -> ModelEntry:
        key = (model_id, task)
        async with self._load_semaphore:
            with _LOCK:
                existing = self._models.get(key)
                if existing and existing.status == "ready":
                    return existing
                if len(self._models) >= self._max_loaded or RESOURCES.need_eviction():
                    self._evict_one()
                runner_cls = get_runner_cls(task)
                dtype_str = choose_dtype(param_count=None, task=task)
                try:
                    runner = runner_cls(model_id=model_id, device=self._device, dtype=dtype_str)  # type: ignore[call-arg]
                except TypeError:
                    # Backwards compatibility for runners that do not yet accept dtype
                    runner = runner_cls(model_id=model_id, device=self._device)
                entry = ModelEntry(model_id=model_id, task=task, runner=runner, status="loading")
                entry.load_started_at = time.time()
                self._models[key] = entry
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = self._loop

            def _load_runner() -> Optional[int]:
                # Wrapper so we can swap to ONNX fallback on specific text-generation errors
                try:
                    return runner.load()
                except Exception as e:  # local fallback logic, mirrors sync path
                    msg = str(e)
                    if task == "text-generation" and "does not appear to have a file named" in msg:
                        from app.core.runners.text_onnx import OnnxTextGenerationRunner
                        onnx_runner = OnnxTextGenerationRunner(model_id=model_id, device=self._device)
                        # replace runner in entry so predict() uses ONNX
                        entry.runner = onnx_runner
                        return onnx_runner.load()
                    raise

            try:
                # Offload potentially heavy runner.load() to thread pool
                param_count = await loop.run_in_executor(None, _load_runner)
                with _LOCK:
                    entry.status = "ready"
                    entry.param_count = param_count
                    entry.load_finished_at = time.time()
                    entry.load_time_ms = int((entry.load_finished_at - entry.load_started_at) * 1000)
                    if param_count:
                        bytes_per_param = 2 if getattr(entry.runner, "dtype", None) in ("float16", "half") else 4
                        entry.mem_estimate_mb = round((param_count * bytes_per_param) / (1024**2), 2)
                return entry
            except Exception as e:
                with _LOCK:
                    entry.load_finished_at = time.time()
                    entry.load_time_ms = int((entry.load_finished_at - (entry.load_started_at or entry.load_finished_at)) * 1000)
                    msg = str(e)
                    entry.status = "error"
                    entry.last_error = msg[:200]
                raise RuntimeError(f"Failed loading model {model_id} for task {task}: {e}")

    def _sync_load_model(self, task: str, model_id: str) -> ModelEntry:
        key = (model_id, task)
        with _LOCK:
            existing = self._models.get(key)
            if existing:
                return existing
            if len(self._models) >= self._max_loaded or RESOURCES.need_eviction():
                self._evict_one()
            runner_cls = get_runner_cls(task)
            dtype_str = choose_dtype(param_count=None, task=task)
            try:
                runner = runner_cls(model_id=model_id, device=self._device, dtype=dtype_str)  # type: ignore[call-arg]
            except TypeError:
                runner = runner_cls(model_id=model_id, device=self._device)
            entry = ModelEntry(model_id=model_id, task=task, runner=runner, status="loading")
            entry.load_started_at = time.time()
            self._models[key] = entry
        try:
            try:
                param_count = runner.load()
            except Exception as e:
                msg = str(e)
                if task == "text-generation" and "does not appear to have a file named" in msg:
                    from app.core.runners.text_onnx import OnnxTextGenerationRunner
                    onnx_runner = OnnxTextGenerationRunner(model_id=model_id, device=self._device)
                    entry.runner = onnx_runner
                    param_count = onnx_runner.load()
                else:
                    raise
            with _LOCK:
                entry.status = "ready"
                entry.param_count = param_count
                entry.load_finished_at = time.time()
                entry.load_time_ms = int((entry.load_finished_at - entry.load_started_at) * 1000)
                if param_count:
                    bytes_per_param = 2 if getattr(entry.runner, "dtype", None) in ("float16", "half") else 4
                    entry.mem_estimate_mb = round((param_count * bytes_per_param) / (1024**2), 2)
                return entry
        except Exception as e:
            with _LOCK:
                entry.load_finished_at = time.time()
                entry.load_time_ms = int((entry.load_finished_at - (entry.load_started_at or entry.load_finished_at)) * 1000)
                msg = str(e)
                entry.status = "error"
                entry.last_error = msg[:200]
            raise RuntimeError(f"Failed loading model {model_id} for task {task}: {e}")

    def _evict_one(self) -> None:
        # Weighted eviction: prefer evicting erroring, old, and large models
        now = time.time()
        best_key = None
        best_score = float("-inf")
        for key, entry in self._models.items():
            if entry.status == "loading":
                continue
            age = now - entry.last_used_at
            size = entry.mem_estimate_mb or 0.0
            error_flag = 1.0 if entry.status == "error" or entry.last_error else 0.0
            score = age + 0.1 * size + 3600.0 * error_flag
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is not None:
            try:
                self._models[best_key].runner.unload()
            except Exception:
                pass
            del self._models[best_key]


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
