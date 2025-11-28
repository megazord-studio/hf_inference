import logging
import os
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fastapi import APIRouter
from fastapi import Query
from fastapi import Response
from fastapi import status
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from pydantic import BaseModel

from app.config import HUB_LIST_LIMIT
from app.config import MODEL_ENRICH_BATCH_LIMIT

router = APIRouter(prefix="/api", tags=["models"])
log = logging.getLogger("app.models")


class ModelSummary(BaseModel):
    id: str
    pipeline_tag: Optional[str] = None
    tags: Optional[List[str]] = None
    gated: Optional[bool] = None
    likes: Optional[int] = None
    downloads: Optional[int] = None
    cardData: Optional[dict] = None


class ModelEnriched(ModelSummary):
    config: Optional[dict] = None
    siblings: Optional[List[dict]] = None
    cardData: Optional[dict] = None  # override type for clarity


class TaskStats(BaseModel):
    task: str
    model_count: int
    total_likes: int
    avg_likes: float
    total_downloads: int
    avg_downloads: float


class ModelMetaLite(BaseModel):
    id: str
    gated: bool = False


# Curated TASKS: broad coverage while pruning niche / low-signal tasks.
# Inclusion heuristics applied offline: (model_count >= 50) OR (avg_likes >= ~40) OR (core user goal) excluding experimental / niche.
# Duplicates unified (e.g. mask-generation folded into image-segmentation; image-feature-extraction into feature-extraction; unconditional-image-generation folded into text-to-image).
TASKS = {
    # Language core & transformation
    "text-generation",
    "summarization",
    "translation",
    "question-answering",
    "table-question-answering",
    "text-classification",
    "token-classification",
    "zero-shot-classification",
    "fill-mask",
    "text-ranking",
    # Embeddings / similarity
    "sentence-similarity",
    "feature-extraction",
    # Vision understanding & detection
    "image-classification",
    "object-detection",
    "image-segmentation",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
    "keypoint-detection",
    "depth-estimation",
    # Image generation & transformation
    "text-to-image",
    "image-to-image",
    "image-super-resolution",
    "image-restoration",
    # 3D / emerging vision (kept due to growth & avg likes)
    "image-to-3d",
    "text-to-3d",
    # Video tasks (generation & conversion)
    "text-to-video",
    "image-to-video",
    # Audio & speech
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-classification",
    "audio-to-audio",
    "text-to-audio",
    "audio-text-to-text",
    "voice-activity-detection",
    # Time series & forecasting
    "time-series-forecasting",
    # Retrieval / multimodal search
    "visual-document-retrieval",
    # High-engagement generalist (kept despite broad scope)
    "any-to-any",
}

SUPPORTED_TASKS = {
    # Phase 0/1 tasks retained elsewhere
    # Vision & Audio Phase 2 (multimodal removed):
    "image-classification",
    "image-captioning",
    "image-to-text",  # keep standard HF alias for captioning
    "object-detection",
    "image-segmentation",
    "depth-estimation",
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-classification",
}

# In-memory cache + TTL (synced with persistent disk cache TTL)
FOUR_DAYS = 4 * 24 * 60 * 60
_CACHE_TTL = FOUR_DAYS
_CACHE: List[ModelSummary] = []
_CACHE_TS: float = 0.0

# Persistent cache path (project root ./cache)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "models_preloaded.json")
CACHE_MODE = "all"  # we now store all pipeline tags

PRELOAD_LIMIT = 30000

# --- Enrich cache (disk-backed) ---
ENRICH_CACHE_FILE = os.path.join(CACHE_DIR, "enrich_cache.json")
_ENRICH_CACHE_TTL = FOUR_DAYS
# Structure: { model_id: { "gated": bool, "ts": float } }
_ENRICH_CACHE: Dict[str, Dict[str, float]] = {}
_ENRICH_TS: float = 0.0


def _ensure_cache_dir() -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as e:
        log.warning(f"Could not create cache directory {CACHE_DIR}: {e}")


def _persist_cache_to_disk() -> None:
    if not _CACHE:
        return
    _ensure_cache_dir()
    tmp_path = CACHE_FILE + ".tmp"
    try:
        payload = {
            "timestamp": _CACHE_TS,
            "mode": CACHE_MODE,
            "models": [m.model_dump() for m in _CACHE],
        }
        import json

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, CACHE_FILE)
        log.info(
            f"Persisted models cache to disk: mode={CACHE_MODE} ts={_CACHE_TS} entries={len(_CACHE)}"
        )
    except Exception as e:
        log.warning(f"Failed to persist models cache: {e}")


def _load_cache_from_disk() -> None:
    global _CACHE, _CACHE_TS
    if not os.path.exists(CACHE_FILE):
        return
    try:
        import json

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", 0.0)
        mode = data.get("mode")
        models_raw = data.get("models", [])
        if not isinstance(models_raw, list):
            return
        # TTL check
        if time.time() - ts > _CACHE_TTL:
            log.info("Disk cache expired; ignoring cached models.")
            return
        _CACHE = [ModelSummary(**m) for m in models_raw]
        _CACHE_TS = ts
        # If legacy cache (no mode or not 'all'), mark for refetch lazily
        if mode != CACHE_MODE:
            log.info(
                f"Legacy cache mode={mode}; will refresh on next full request."
            )
        else:
            log.info(
                f"Loaded models cache from disk: mode={mode} entries={len(_CACHE)} age={int(time.time() - _CACHE_TS)}s"
            )
    except Exception as e:
        log.warning(f"Failed to load models cache from disk: {e}")


# --- Enrich cache helpers ---


def _load_enrich_cache_from_disk() -> None:
    global _ENRICH_CACHE, _ENRICH_TS
    if not os.path.exists(ENRICH_CACHE_FILE):
        return
    try:
        import json

        with open(ENRICH_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", 0.0)
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return
        # TTL check for whole cache
        if time.time() - ts > _ENRICH_CACHE_TTL:
            log.info("Enrich disk cache expired; ignoring cached entries.")
            return
        _ENRICH_CACHE = entries
        _ENRICH_TS = ts
        log.info(
            f"Loaded enrich cache from disk: entries={len(_ENRICH_CACHE)} age={int(time.time() - _ENRICH_TS)}s"
        )
    except Exception as e:
        log.warning(f"Failed to load enrich cache from disk: {e}")


def _persist_enrich_cache_to_disk() -> None:
    _ensure_cache_dir()
    tmp = ENRICH_CACHE_FILE + ".tmp"
    try:
        import json

        payload = {
            "timestamp": _ENRICH_TS,
            "entries": _ENRICH_CACHE,
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, ENRICH_CACHE_FILE)
        log.info(
            f"Persisted enrich cache to disk: entries={len(_ENRICH_CACHE)} ts={_ENRICH_TS}"
        )
    except Exception as e:
        log.warning(f"Failed to persist enrich cache: {e}")


def _fetch_models(limit: int) -> List[ModelSummary]:
    """Fetch models from HF Hub with explicit limit.

    Args:
        limit: Maximum number of models to fetch (capped by HUB_LIST_LIMIT config)

    Returns:
        List of ModelSummary objects
    """
    api = HfApi()
    results: List[ModelSummary] = []
    # Apply explicit limit cap from config
    effective_limit = min(limit, HUB_LIST_LIMIT)
    try:
        iterator = api.list_models(
            sort="downloads", direction=-1, limit=effective_limit
        )
        for info in iterator:
            mid = getattr(info, "modelId", None)
            if not mid:
                continue
            pt = getattr(info, "pipeline_tag", None)
            results.append(
                ModelSummary(
                    id=mid,
                    pipeline_tag=pt,
                    tags=getattr(info, "tags", None),
                    gated=getattr(info, "gated", None),
                    likes=getattr(info, "likes", None),
                    downloads=getattr(info, "downloads", None),
                )
            )
            if len(results) >= effective_limit:
                break
    except HfHubHTTPError as e:  # network or rate-limit errors
        raise RuntimeError(f"Failed to list models from HF Hub: {e}")
    return results


def _prewarm_cache() -> None:
    """Populate in-memory + disk cache at import time for faster first request.

    Skips if a valid disk cache is present and fresh. Fetches a larger superset
    controlled by PRELOAD_LIMIT.
    """
    global _CACHE, _CACHE_TS
    if _CACHE:
        return
    _load_cache_from_disk()
    if _CACHE and (time.time() - _CACHE_TS) < _CACHE_TTL:
        return  # already warm and fresh
    log.info(f"Prewarming models cache (limit={PRELOAD_LIMIT}) ...")
    _CACHE = _fetch_models(PRELOAD_LIMIT)
    _CACHE_TS = time.time()
    _persist_cache_to_disk()
    log.info(f"Prewarm complete: entries={len(_CACHE)}")


_prewarm_cache()


@router.get("/models/preloaded", response_model=List[ModelSummary])
async def get_preloaded_models(
    limit: int = Query(1000, ge=1, le=50000),
    refresh: bool = Query(False, description="Force refresh of cached set"),
) -> List[ModelSummary]:
    global _CACHE, _CACHE_TS
    now = time.time()
    if not refresh and not _CACHE:
        _load_cache_from_disk()
    need_refresh = (
        refresh
        or not _CACHE
        or (now - _CACHE_TS) > _CACHE_TTL
        or len(_CACHE) < limit
    )
    if need_refresh:
        target = max(limit, PRELOAD_LIMIT)
        _CACHE = _fetch_models(target)
        _CACHE_TS = time.time()
        _persist_cache_to_disk()
        log.info(f"Refreshed preloaded models cache: size={len(_CACHE)}")
    return _CACHE[:limit]


@router.get("/models/preloaded/meta")
async def preloaded_meta() -> Dict[str, Any]:
    return {
        "cached": len(_CACHE),
        "last_refresh_ts": _CACHE_TS,
        "ttl": _CACHE_TTL,
        "disk_file": CACHE_FILE,
        "disk_exists": os.path.exists(CACHE_FILE),
        "preload_limit": PRELOAD_LIMIT,
        "tasks_count": len(TASKS),
    }


def _compute_task_stats(
    task_ids: List[str], include_empty: bool, min_count: int
) -> List[TaskStats]:
    if not _CACHE:
        _load_cache_from_disk()
    stats: List[TaskStats] = []
    for task in sorted(task_ids):
        models = [m for m in _CACHE if m.pipeline_tag == task]
        if not models and not include_empty:
            continue
        mc = len(models)
        if mc < min_count:
            continue
        tot_likes = sum((m.likes or 0) for m in models)
        tot_downloads = sum((m.downloads or 0) for m in models)
        avg_likes = (tot_likes / mc) if mc else 0.0
        avg_downloads = (tot_downloads / mc) if mc else 0.0
        stats.append(
            TaskStats(
                task=task,
                model_count=mc,
                total_likes=tot_likes,
                avg_likes=round(avg_likes, 2),
                total_downloads=tot_downloads,
                avg_downloads=round(avg_downloads, 2),
            )
        )
    return stats


@router.get("/models/tasks", response_model=List[TaskStats])
async def list_tasks(
    scope: str = Query(
        "relevant",
        pattern="^(relevant|all)$",
        description="Task scope: curated TASKS or all pipeline_tags",
    ),
    sort_by: str = Query(
        "models",
        pattern="^(models|likes|downloads)$",
        description="Primary sort field",
    ),
    include_empty: bool = Query(
        False,
        description="Include tasks with zero models (only relevant scope)",
    ),
    min_count: int = Query(
        0, ge=0, description="Filter out tasks below this model count"
    ),
    secondary_sort_desc: bool = Query(
        True, description="Sort descending for primary field"
    ),
    force_full_refresh: bool = Query(
        False, description="Force a full cache refresh (scope=all only)"
    ),
) -> List[TaskStats]:
    global _CACHE, _CACHE_TS
    if scope == "all":
        legacy_cache = os.path.exists(CACHE_FILE) and not any(
            getattr(m, "pipeline_tag", None) for m in _CACHE
        )
        if force_full_refresh or legacy_cache:
            log.info("Full refresh triggered for all-task scope.")
            _CACHE = _fetch_models(PRELOAD_LIMIT)
            _CACHE_TS = time.time()
            _persist_cache_to_disk()
        task_ids = sorted({m.pipeline_tag for m in _CACHE if m.pipeline_tag})
        stats = _compute_task_stats(
            task_ids, include_empty=False, min_count=min_count
        )
    else:
        task_ids = sorted(TASKS)
        stats = _compute_task_stats(
            task_ids, include_empty=include_empty, min_count=min_count
        )
    reverse = secondary_sort_desc
    if sort_by == "models":
        stats.sort(
            key=lambda s: (s.model_count, s.total_downloads, s.total_likes),
            reverse=reverse,
        )
    elif sort_by == "likes":
        stats.sort(
            key=lambda s: (s.total_likes, s.model_count), reverse=reverse
        )
    elif sort_by == "downloads":
        stats.sort(
            key=lambda s: (s.total_downloads, s.model_count), reverse=reverse
        )
    return stats


def _enrich_single_model(model_id: str) -> ModelMetaLite:
    """Fetch minimal metadata for a single model id (only `gated`).

    Always returns a boolean for `gated` (True for gated, False otherwise).
    Uses model_info (no expand) to avoid heavy timeouts; falls back to a
    lightweight list_models search if needed. Designed for small UI batches.
    """
    api = HfApi()
    try:
        info = api.model_info(model_id)
        gv = getattr(info, "gated", None)
        return ModelMetaLite(
            id=model_id,
            gated=bool(gv),
        )
    except Exception:
        try:
            for m in api.list_models(search=f"id={model_id}", limit=1):
                gv = getattr(m, "gated", None)
                return ModelMetaLite(
                    id=model_id,
                    gated=bool(gv),
                )
        except Exception as e2:
            log.debug(
                f"enrich_single_model fallback failed for {model_id}: {e2}"
            )
    return ModelMetaLite(id=model_id, gated=False)


@router.post("/models/enrich", response_model=List[ModelMetaLite])
async def enrich_models(
    models: List[str],
    since_ts: float = Query(
        0.0, description="Client's last known enrich cache timestamp"
    ),
    force_refresh: bool = Query(
        False, description="Force refresh of enrich cache"
    ),
) -> Union[Response, JSONResponse]:
    """Enrich a small list of model ids with only the `gated` flag for the UI.

    Behavior changes:
    - Uses a disk-backed per-model cache at ./cache/enrich_cache.json to avoid
      repeated HF Hub calls.
    - Only queries HF for models missing from the cache or expired.
    - If the client provides `since_ts` and there are no updates newer than
      that timestamp, returns 204 No Content so the frontend doesn't need to
      re-render.
    """
    global _ENRICH_CACHE, _ENRICH_TS
    max_batch = MODEL_ENRICH_BATCH_LIMIT
    ids = models[:max_batch]
    now = time.time()

    # Load enrich cache lazily
    if not _ENRICH_CACHE:
        _load_enrich_cache_from_disk()

    # Determine which ids need fetching
    to_fetch: List[str] = []
    for mid in ids:
        entry = _ENRICH_CACHE.get(mid)
        if force_refresh:
            to_fetch.append(mid)
            continue
        if not entry:
            to_fetch.append(mid)
            continue
        # entry is expected to have a ts field
        entry_ts = float(entry.get("ts", 0.0))
        if now - entry_ts > _ENRICH_CACHE_TTL:
            to_fetch.append(mid)

    # Fetch missing/expired entries
    updated = False
    for mid in to_fetch:
        meta = _enrich_single_model(mid)
        _ENRICH_CACHE[mid] = {"gated": bool(meta.gated), "ts": time.time()}
        updated = True

    if updated:
        # bump global enrich cache timestamp to now and persist
        _ENRICH_TS = time.time()
        _persist_enrich_cache_to_disk()

    # If client is up-to-date and nothing changed, return 204 No Content
    if not updated and since_ts and _ENRICH_TS and _ENRICH_TS <= since_ts:
        # include cache ts so the client still knows the current cache timestamp
        return Response(
            status_code=status.HTTP_204_NO_CONTENT,
            headers={"X-Enrich-Cache-Ts": str(_ENRICH_TS or now)},
        )

    # Build response list in same order as request
    metas: List[ModelMetaLite] = []
    for mid in ids:
        entry = _ENRICH_CACHE.get(mid)
        if entry:
            metas.append(
                ModelMetaLite(id=mid, gated=bool(entry.get("gated", False)))
            )
        else:
            # fallback: if something odd happened, call the fetch function synchronously
            metas.append(_enrich_single_model(mid))

    # Return proper JSONResponse so Content-Length is set correctly
    try:
        body = [m.model_dump() for m in metas]
        return JSONResponse(
            content=body,
            status_code=status.HTTP_200_OK,
            headers={"X-Enrich-Cache-Ts": str(_ENRICH_TS or now)},
        )
    except Exception:
        # As a last-resort fallback, let FastAPI serialize the Pydantic models list
        return JSONResponse(
            content=[m.model_dump() for m in metas],
            status_code=status.HTTP_200_OK,
            headers={"X-Enrich-Cache-Ts": str(_ENRICH_TS or now)},
        )
