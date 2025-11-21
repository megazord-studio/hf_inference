import os
import time
from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from huggingface_hub import HfApi
import logging

router = APIRouter(prefix="/api", tags=["models"])
log = logging.getLogger("uvicorn.error")

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

# Curated TASKS: broad coverage while pruning niche / low-signal tasks.
# Inclusion heuristics applied offline: (model_count >= 50) OR (avg_likes >= ~40) OR (core user goal) excluding experimental / niche.
# Duplicates unified (e.g. mask-generation folded into image-segmentation; image-feature-extraction into feature-extraction; unconditional-image-generation folded into text-to-image).
TASKS = {
    # Language core & transformation
    "text-generation","summarization","translation","question-answering","table-question-answering",
    "text-classification","token-classification","zero-shot-classification","fill-mask","text-ranking",
    # Embeddings / similarity
    "sentence-similarity","feature-extraction",
    # Vision understanding & detection
    "image-classification","object-detection","image-segmentation","zero-shot-image-classification","zero-shot-object-detection","keypoint-detection","depth-estimation",
    # Image generation & transformation
    "text-to-image","image-to-image","image-super-resolution","image-restoration",
    # 3D / emerging vision (kept due to growth & avg likes)
    "image-to-3d","text-to-3d",
    # Video tasks (generation & conversion)
    "text-to-video","image-to-video",
    # Audio & speech
    "automatic-speech-recognition","text-to-speech","audio-classification","audio-to-audio","text-to-audio","audio-text-to-text","voice-activity-detection",
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "models_preloaded.json")
CACHE_MODE = "all"  # we now store all pipeline tags

PRELOAD_LIMIT = int(os.getenv("PRELOAD_MODELS_LIMIT", "30000"))  # configurable preload size


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
        log.info(f"Persisted models cache to disk: mode={CACHE_MODE} ts={_CACHE_TS} entries={len(_CACHE)}")
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
            log.info(f"Legacy cache mode={mode}; will refresh on next full request.")
        else:
            log.info(f"Loaded models cache from disk: mode={mode} entries={len(_CACHE)} age={int(time.time()-_CACHE_TS)}s")
    except Exception as e:
        log.warning(f"Failed to load models cache from disk: {e}")


def _fetch_models(limit: int) -> List[ModelSummary]:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    results: List[ModelSummary] = []
    try:
        for info in api.list_models(sort="downloads", direction=-1):  # type: ignore
            mid = getattr(info, "modelId", None)
            if not mid:
                continue
            pt = getattr(info, "pipeline_tag", None)
            results.append(ModelSummary(
                id=mid,
                pipeline_tag=pt,
                tags=getattr(info, "tags", None),
                gated=getattr(info, "gated", None),
                likes=getattr(info, "likes", None),
                downloads=getattr(info, "downloads", None),
            ))
            if len(results) >= limit:
                break
    except Exception as e:  # network or rate-limit errors
        log.warning(f"Failed to list models from HF Hub: {e}")
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
    refresh: bool = Query(False, description="Force refresh of cached set")
) -> List[ModelSummary]:
    global _CACHE, _CACHE_TS
    now = time.time()
    if not refresh and not _CACHE:
        _load_cache_from_disk()
    need_refresh = refresh or not _CACHE or (now - _CACHE_TS) > _CACHE_TTL or len(_CACHE) < limit
    if need_refresh:
        target = max(limit, PRELOAD_LIMIT)
        _CACHE = _fetch_models(target)
        _CACHE_TS = time.time()
        _persist_cache_to_disk()
        log.info(f"Refreshed preloaded models cache: size={len(_CACHE)}")
    return _CACHE[:limit]

@router.get("/models/preloaded/meta")
async def preloaded_meta():
    return {
        "cached": len(_CACHE),
        "last_refresh_ts": _CACHE_TS,
        "ttl": _CACHE_TTL,
        "disk_file": CACHE_FILE,
        "disk_exists": os.path.exists(CACHE_FILE),
        "preload_limit": PRELOAD_LIMIT,
        "tasks_count": len(TASKS),
    }

def _compute_task_stats(task_ids: List[str], include_empty: bool, min_count: int) -> List[TaskStats]:
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
        stats.append(TaskStats(
            task=task,
            model_count=mc,
            total_likes=tot_likes,
            avg_likes=round(avg_likes, 2),
            total_downloads=tot_downloads,
            avg_downloads=round(avg_downloads, 2),
        ))
    return stats

@router.get("/models/tasks", response_model=List[TaskStats])
async def list_tasks(
    scope: str = Query("relevant", pattern="^(relevant|all)$", description="Task scope: curated TASKS or all pipeline_tags"),
    sort_by: str = Query("models", pattern="^(models|likes|downloads)$", description="Primary sort field"),
    include_empty: bool = Query(False, description="Include tasks with zero models (only relevant scope)"),
    min_count: int = Query(0, ge=0, description="Filter out tasks below this model count"),
    secondary_sort_desc: bool = Query(True, description="Sort descending for primary field"),
    force_full_refresh: bool = Query(False, description="Force a full cache refresh (scope=all only)")
) -> List[TaskStats]:
    global _CACHE, _CACHE_TS
    if scope == 'all':
        legacy_cache = os.path.exists(CACHE_FILE) and not any(getattr(m, 'pipeline_tag', None) for m in _CACHE)
        if force_full_refresh or legacy_cache:
            log.info("Full refresh triggered for all-task scope.")
            _CACHE = _fetch_models(PRELOAD_LIMIT)
            _CACHE_TS = time.time()
            _persist_cache_to_disk()
        task_ids = sorted({m.pipeline_tag for m in _CACHE if m.pipeline_tag})
        stats = _compute_task_stats(task_ids, include_empty=False, min_count=min_count)
    else:
        task_ids = sorted(TASKS)
        stats = _compute_task_stats(task_ids, include_empty=include_empty, min_count=min_count)
    reverse = secondary_sort_desc
    if sort_by == 'models':
        stats.sort(key=lambda s: (s.model_count, s.total_downloads, s.total_likes), reverse=reverse)
    elif sort_by == 'likes':
        stats.sort(key=lambda s: (s.total_likes, s.model_count), reverse=reverse)
    elif sort_by == 'downloads':
        stats.sort(key=lambda s: (s.total_downloads, s.model_count), reverse=reverse)
    return stats
