from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

from app.config import REGISTRY_MAX_LOADED_MODELS
from app.config import REGISTRY_MEMORY_LIMIT_MB
from app.core.device import choose_dtype
from app.core.device import select_device
from app.core.runners import get_runner_cls

log = logging.getLogger("app.registry")

DEFAULT_MAX_LOADED = REGISTRY_MAX_LOADED_MODELS
DEFAULT_MEMORY_LIMIT_MB = REGISTRY_MEMORY_LIMIT_MB
ENV_MAX_LOADED = None
ENV_MEMORY_LIMIT_MB = None


def _read_positive_int(
    env_name: str, default: Optional[int], *, allow_none: bool = False
) -> Optional[int]:
    value = os.getenv(env_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        log.warning(
            "invalid %s value %r; defaulting to %s", env_name, value, default
        )
        return default
    if parsed <= 0:
        return None if allow_none else default
    return parsed


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
    popularity_score: Optional[float] = (
        None  # placeholder (Phase 0 not computed)
    )
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
        # Caps can be tuned via env vars (see _read_positive_int helper).
        # Allow overriding limits via environment for test/runtime tuning.
        # Use distinct env names to avoid collision with other config.
        self._max_loaded = (
            _read_positive_int("APP_REGISTRY_MAX_LOADED", DEFAULT_MAX_LOADED)
            or DEFAULT_MAX_LOADED
        )
        # Soft memory cap (MB) to prevent OOM from repeated loads. Passing 0 or
        # a negative value disables the memory limit (returns None when allow_none=True).
        self._memory_limit_mb = _read_positive_int(
            "APP_REGISTRY_MEMORY_LIMIT_MB",
            DEFAULT_MEMORY_LIMIT_MB,
            allow_none=True,
        )
        # Create an event loop for background async loads and start the thread.
        # Keep the loop object on `self` so other methods can submit coroutines.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._loop_thread.start()
        # Wait briefly for the background event loop to start to avoid races
        # where run_coroutine_threadsafe() is called before the loop is running.
        for _ in range(100):
            if self._loop.is_running():
                break
            time.sleep(0.01)
        else:
            log.warning("background asyncio loop did not start within timeout")
        self._loading_futures: Dict[
            Tuple[str, str], Union[asyncio.Future[Any], Future[Any], None]
        ] = {}
        # Limit number of concurrent heavy loads
        self._max_concurrent_loads = 2
        self._load_semaphore = asyncio.Semaphore(self._max_concurrent_loads)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # --- public API ---
    def predict(
        self,
        task: str,
        model_id: str,
        inputs: Dict[str, Any],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Ensure model is loaded (may block for heavy models)
        entry = self._get_or_load(task, model_id)
        # touch last-used
        with _LOCK:
            entry.touch()

        # Run predict on the runner; keep it simple and synchronous for tests
        start = time.time()
        try:
            # delegate to runner.predict; allow runner to raise which we'll surface
            output = entry.runner.predict(inputs, options or {})
        except Exception as e:
            # Log full traceback for debugging and store a descriptive last_error
            log.exception("runner.predict failed for %s:%s", model_id, task)
            with _LOCK:
                entry.status = "error"
                entry.last_error = repr(e)[:1000]
            # Raise a RuntimeError with repr to ensure the API surfaces a non-empty message
            raise RuntimeError(repr(e))
        finally:
            runtime_ms = int((time.time() - start) * 1000)

        # Update last_used timestamp
        with _LOCK:
            entry.last_used_at = time.time()

        result: Dict[str, Any] = {
            "output": output,
            "runtime_ms": runtime_ms,
            "resolved_model_id": model_id,
        }
        # Add backend metadata if runner provides it
        backend = getattr(entry.runner, "backend", None)
        if backend is not None:
            result["backend"] = backend
        return result

    def _total_mem_used_mb(self) -> float:
        # Sum up known estimates; unknown entries count as 0
        with _LOCK:
            return sum(
                (e.mem_estimate_mb or 0.0) for e in self._models.values()
            )

    def _compute_eviction_score(self, entry: ModelEntry, now: float) -> float:
        # Higher score = more eligible for eviction
        age = now - entry.last_used_at
        size = entry.mem_estimate_mb or 0.0
        error_flag = (
            1.0 if entry.status == "error" or entry.last_error else 0.0
        )
        popularity = entry.popularity_score or 0.0
        # Combine signals: prioritize error and long-unused; penalize popular models
        return age + 0.1 * size + 3600.0 * error_flag - 100.0 * popularity

    def list_loaded(self) -> Dict[str, Any]:  # debug/inspection
        with _LOCK:
            now = time.time()
            return {
                f"{mid}:{task}": {
                    "status": e.status,
                    "last_used_s_ago": int(now - e.last_used_at),
                    "loaded_s_ago": int(now - e.loaded_at)
                    if e.loaded_at
                    else None,
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
                key
                for key in self._models.keys()
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

    def reset(self) -> None:
        with _LOCK:
            for entry in self._models.values():
                try:
                    entry.runner.unload()
                except Exception:
                    pass
            self._models.clear()
            self._loading_futures.clear()

    # --- internals ---
    def _get_or_load(self, task: str, model_id: str) -> ModelEntry:
        key = (model_id, task)
        with _LOCK:
            existing = self._models.get(key)
            if existing:
                return existing
            # Do not evict here; eviction will be performed inside the
            # concrete load functions immediately before creating a runner.

            if task in _HEAVY_TASKS:
                # Use async loading queue for heavy models, blocking caller only while waiting
                future = self._loading_futures.get(key)
                if future is None or (
                    hasattr(future, "done") and future.done()
                ):
                    future = asyncio.run_coroutine_threadsafe(
                        self._async_load_model(task, model_id), self._loop
                    )
                    self._loading_futures[key] = future
                runner_entry_future: Any = self._loading_futures.get(key)
            else:
                runner_entry_future = None

        if task in _HEAVY_TASKS and runner_entry_future is not None:
            # Block current thread until model is ready or error raised
            return runner_entry_future.result()

        # For non-heavy tasks, fall back to sync loading
        return self._sync_load_model(task, model_id)

    async def _async_load_model(self, task: str, model_id: str) -> ModelEntry:
        key = (model_id, task)
        async with self._load_semaphore:
            with _LOCK:
                existing = self._models.get(key)
                if existing and existing.status == "ready":
                    return existing
                # Do not evict pre-emptively here; evict after the new model finishes loading
                runner_cls = get_runner_cls(task)
                dtype_str = choose_dtype(param_count=None, task=task)
                try:
                    runner = runner_cls(
                        model_id=model_id, device=self._device, dtype=dtype_str
                    )  # type: ignore[call-arg]
                except TypeError:
                    runner = runner_cls(model_id=model_id, device=self._device)
                entry = ModelEntry(
                    model_id=model_id,
                    task=task,
                    runner=runner,
                    status="loading",
                )
                entry.load_started_at = time.time()
                self._models[key] = entry
                log.info(
                    "model registered loading %s:%s (count=%d)",
                    model_id,
                    task,
                    len(self._models),
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = self._loop

            # Offload potentially heavy runner.load() to thread pool
            def _load_runner() -> int:
                # local wrapper to allow swap to ONNX fallback on specific errors
                try:
                    return entry.runner.load()
                except Exception as e:
                    msg = str(e)
                    if task == "text-generation":
                        try:
                            from app.core.runners.text_onnx import (
                                OnnxTextGenerationRunner,
                            )

                            entry.runner = OnnxTextGenerationRunner(
                                model_id=model_id, device=self._device
                            )
                            return entry.runner.load()
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed loading model {model_id} for task {task}: {msg}"
                            ) from e2
                    raise RuntimeError(
                        f"Failed loading model {model_id} for task {task}: {msg}"
                    )

            try:
                param_count = await loop.run_in_executor(None, _load_runner)
                with _LOCK:
                    entry.status = "ready"
                    entry.param_count = param_count
                    entry.load_finished_at = time.time()
                    entry.load_time_ms = int(
                        (entry.load_finished_at - entry.load_started_at) * 1000
                    )
                    if param_count:
                        bytes_per_param = (
                            2
                            if getattr(entry.runner, "dtype", None)
                            in ("float16", "half")
                            else 4
                        )
                        # Use integer megabyte estimates to keep totals deterministic in tests
                        entry.mem_estimate_mb = int(
                            (param_count * bytes_per_param) / (1024**2)
                        )
                    # Ensure runner is marked loaded so BaseRunner.predict() does not raise
                    try:
                        setattr(entry.runner, "_loaded", True)
                    except Exception:
                        pass
                # After a successful load, evict if we are over the count limit.
                # Eviction prefers older/less-used models via _compute_eviction_score,
                # so the freshly-loaded entry is unlikely to be chosen.
                self._evict_to_limit(exclude_keys={key})
                log.info(
                    "model ready %s:%s (mem_mb=%s) loaded, total=%d",
                    model_id,
                    task,
                    entry.mem_estimate_mb,
                    len(self._models),
                )
                return entry
            except Exception as e:
                with _LOCK:
                    entry.load_finished_at = time.time()
                    entry.load_time_ms = int(
                        (
                            entry.load_finished_at
                            - (entry.load_started_at or entry.load_finished_at)
                        )
                        * 1000
                    )
                    msg = str(e)
                    entry.status = "error"
                    entry.last_error = msg[:200]
                raise RuntimeError(
                    f"Failed loading model {model_id} for task {task}: {e}"
                )

    def _sync_load_model(self, task: str, model_id: str) -> ModelEntry:
        key = (model_id, task)
        with _LOCK:
            existing = self._models.get(key)
            if existing:
                return existing
            # Do not evict pre-emptively here; evict after the new model finishes loading
            runner_cls = get_runner_cls(task)
            dtype_str = choose_dtype(param_count=None, task=task)
            try:
                runner = runner_cls(
                    model_id=model_id, device=self._device, dtype=dtype_str
                )  # type: ignore[call-arg]
            except TypeError:
                runner = runner_cls(model_id=model_id, device=self._device)
            entry = ModelEntry(
                model_id=model_id, task=task, runner=runner, status="loading"
            )
            entry.load_started_at = time.time()
            self._models[key] = entry
            log.info(
                "model registered loading %s:%s (count=%d)",
                model_id,
                task,
                len(self._models),
            )
        try:
            try:
                param_count = runner.load()
            except Exception as e:
                msg = str(e)
                # Fallback: for text-generation, try ONNX runner when weights are missing
                if task == "text-generation":
                    try:
                        from app.core.runners.text_onnx import (
                            OnnxTextGenerationRunner,
                        )

                        runner = OnnxTextGenerationRunner(
                            model_id=model_id, device=self._device
                        )
                        self._models[key].runner = runner
                        param_count = runner.load()
                    except Exception as e2:
                        raise RuntimeError(
                            f"Failed loading model {model_id} for task {task}: {msg}"
                        ) from e2
                else:
                    raise RuntimeError(
                        f"Failed loading model {model_id} for task {task}: {msg}"
                    )
            with _LOCK:
                entry.status = "ready"
                entry.param_count = param_count
                entry.load_finished_at = time.time()
                entry.load_time_ms = int(
                    (entry.load_finished_at - entry.load_started_at) * 1000
                )
                if param_count:
                    bytes_per_param = (
                        2
                        if getattr(entry.runner, "dtype", None)
                        in ("float16", "half")
                        else 4
                    )
                    entry.mem_estimate_mb = int(
                        (param_count * bytes_per_param) / (1024**2)
                    )
                # Ensure runner is marked loaded so BaseRunner.predict() does not raise
                try:
                    setattr(entry.runner, "_loaded", True)
                except Exception:
                    pass
                # After a successful load, evict if we are over the count limit.
                self._evict_to_limit(exclude_keys={key})
                log.info(
                    "model ready %s:%s (mem_mb=%s) loaded, total=%d",
                    model_id,
                    task,
                    entry.mem_estimate_mb,
                    len(self._models),
                )
                return entry
        except Exception as e:
            with _LOCK:
                entry.load_finished_at = time.time()
                entry.load_time_ms = int(
                    (
                        entry.load_finished_at
                        - (entry.load_started_at or entry.load_finished_at)
                    )
                    * 1000
                )
                msg = str(e)
                entry.status = "error"
                entry.last_error = msg[:200]
            raise RuntimeError(
                f"Failed loading model {model_id} for task {task}: {e}"
            )

    def _evict_one(self) -> None:
        # Evict the single most eligible model based on eviction score
        now = time.time()
        best_key = None
        best_score = float("-inf")
        with _LOCK:
            for key, entry in list(self._models.items()):
                if entry.status == "loading":
                    continue
                score = self._compute_eviction_score(entry, now)
                if score > best_score:
                    best_score = score
                    best_key = key
            if best_key is not None:
                mid, mtask = best_key
                try:
                    log.info(
                        "evicting %s:%s score=%.2f mem=%s",
                        mid,
                        mtask,
                        best_score,
                        self._models[best_key].mem_estimate_mb,
                    )
                    self._models[best_key].runner.unload()
                except Exception:
                    log.exception("error unloading model %s:%s", mid, mtask)
                try:
                    del self._models[best_key]
                except Exception:
                    log.exception(
                        "failed deleting model entry %s:%s from registry",
                        mid,
                        mtask,
                    )

    def _evict_to_limit(
        self, exclude_keys: Optional[Iterable[Tuple[str, str]]] = None
    ) -> None:
        """Evict models while either count or estimated memory exceeds limits."""
        protected = set(exclude_keys or ())
        while True:
            with _LOCK:
                now = time.time()
                total_count = len(self._models)
                total_mem = sum(
                    (e.mem_estimate_mb or 0.0) for e in self._models.values()
                )
                mem_limit = self._memory_limit_mb
                over_count = total_count > self._max_loaded
                over_mem = mem_limit is not None and total_mem > mem_limit
                if not over_count and not over_mem:
                    break
                candidates = [
                    (key, entry, self._compute_eviction_score(entry, now))
                    for key, entry in self._models.items()
                    if entry.status != "loading" and key not in protected
                ]
            if not candidates:
                log.debug(
                    "eviction skipped; no candidates but limits exceeded (count=%d/%d mem=%.2f/%s)",
                    total_count,
                    self._max_loaded,
                    total_mem,
                    mem_limit if mem_limit is not None else "inf",
                )
                break
            candidates.sort(
                key=lambda item: (item[2], item[1].mem_estimate_mb or 0.0),
                reverse=True,
            )
            best_key = candidates[0][0]
            with _LOCK:
                entry = self._models.get(best_key)
                if entry is None:
                    continue
                mid, mtask = best_key
                try:
                    log.info(
                        "evicting %s:%s (count=%d/%d mem=%.2f/%s)",
                        mid,
                        mtask,
                        len(self._models),
                        self._max_loaded,
                        sum(
                            (e.mem_estimate_mb or 0.0)
                            for e in self._models.values()
                        ),
                        mem_limit if mem_limit is not None else "inf",
                    )
                    entry.runner.unload()
                except Exception:
                    log.exception("error unloading model %s:%s", mid, mtask)
                finally:
                    try:
                        del self._models[best_key]
                    except Exception:
                        log.exception(
                            "failed deleting model entry %s:%s from registry",
                            mid,
                            mtask,
                        )

    def _ensure_limits(self) -> None:
        self._evict_to_limit()


# mappings and exports (kept at bottom)
PIPELINE_TO_TASK = {
    "automatic-speech-recognition": "automatic-speech-recognition",
    "audio-classification": "audio-classification",
    "text-to-speech": "text-to-speech",
    # Text / NLP pipelines
    "feature-extraction": "embedding",
    "fill-mask": "fill-mask",
    "image-to-text": "image-to-text",
    "question-answering": "question-answering",
    "sentence-similarity": "sentence-similarity",
    "token-classification": "token-classification",
    "text-ranking": "text-ranking",
    "reranking": "text-ranking",
    "translation": "translation",
    "zero-shot-classification": "zero-shot-classification",
    "table-question-answering": "table-question-answering",
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
    "text2text-generation": "summarization",
    "summarization": "summarization",
}

# Singleton instance
REGISTRY = ModelRegistry()

__all__ = ["REGISTRY", "PIPELINE_TO_TASK", "ModelRegistry"]
