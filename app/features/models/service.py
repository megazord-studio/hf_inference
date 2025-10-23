from __future__ import annotations

import datetime
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from urllib.parse import parse_qs
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]

HF_API = "https://huggingface.co/api/models"

# --- Enhanced cache with file persistence (4h TTL) ---------------------------
# Impure by necessity (I/O boundary), but clearly documented and isolated.

_CACHE_TTL = timedelta(hours=4)  # Increased from 10min to 4h
_CACHE_DIR = Path(tempfile.gettempdir()) / "hf_inference_cache"

# Immutable cache entry type
CacheEntry = Tuple[datetime.datetime, List[Dict[str, Any]]]

# In-memory cache (fast path)
_cache_min: Dict[str, CacheEntry] = {}


def _get_cache_file_path(task: str) -> Path:
    """Get cache file path for a task (pure function)."""
    # Use task name and timestamp-based key for cache file
    safe_task = task.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"models_{safe_task}.json"


def _load_from_file_cache(task: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load cache from file if valid (impure I/O operation).
    
    Side effects:
        - Reads from filesystem
    """
    try:
        cache_file = _get_cache_file_path(task)
        if not cache_file.exists():
            return None
        
        # Read cache file
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        timestamp_str = cache_data.get("timestamp")
        data = cache_data.get("data")
        
        if not timestamp_str or data is None:
            return None
        
        # Parse timestamp
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
        
        # Check if still valid
        if (datetime.datetime.now(datetime.UTC) - timestamp) < _CACHE_TTL:
            return data
        
        return None
    except Exception:
        # If any error, just return None (cache miss)
        return None


def _save_to_file_cache(task: str, data: List[Dict[str, Any]]) -> None:
    """
    Save cache to file (impure I/O operation).
    
    Side effects:
        - Writes to filesystem
        - Creates directory if needed
    """
    try:
        # Ensure cache directory exists
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_file = _get_cache_file_path(task)
        cache_data = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "data": data,
        }
        
        # Write atomically (write to temp, then rename)
        temp_file = cache_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(cache_data, f)
        
        # Atomic rename
        temp_file.replace(cache_file)
    except Exception:
        # If save fails, just log and continue (cache is optional)
        pass


def get_cached_min(task: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get cached data for task if valid (impure read operation).
    
    Tries in-memory cache first (fast), then file cache (persistent).
    
    Note: This function is impure as it reads module-level state and filesystem.
    
    Args:
        task: Task identifier
    
    Returns:
        Cached data if valid, None otherwise
    
    Side effects:
        - Reads module-level state
        - Reads from filesystem
    """
    # Fast path: check in-memory cache
    entry = _cache_min.get(task)
    if entry is not None:
        timestamp, data = entry
        if (datetime.datetime.now(datetime.UTC) - timestamp) < _CACHE_TTL:
            return data
    
    # Slow path: check file cache
    file_data = _load_from_file_cache(task)
    if file_data is not None:
        # Update in-memory cache for next time
        _cache_min[task] = (datetime.datetime.now(datetime.UTC), file_data)
        return file_data
    
    return None


def set_cached_min(task: str, data: List[Dict[str, Any]]) -> None:
    """
    Cache data for task (impure write operation).
    
    Saves to both in-memory cache (fast) and file cache (persistent).
    
    Note: This function explicitly mutates module-level state and filesystem.
    
    Args:
        task: Task identifier
        data: Data to cache
    
    Side effects:
        - Mutates _cache_min module-level dict
        - Writes to filesystem
    """
    timestamp = datetime.datetime.now(datetime.UTC)
    _cache_min[task] = (timestamp, data)
    _save_to_file_cache(task, data)


# ----------------------------- helpers ---------------------------------------


def _parse_next_cursor(resp: requests.Response) -> Optional[str]:
    link = resp.headers.get("Link") or resp.headers.get("link")
    if not link:
        return None
    for part in [p.strip() for p in link.split(",")]:
        if 'rel="next"' in part and "<" in part and ">" in part:
            url = part[part.find("<") + 1 : part.find(">")]
            try:
                q = parse_qs(urlparse(url).query)
                return q.get("cursor", [None])[0]
            except Exception:
                return None
    return None


def gated_to_str(val: Any) -> str:
    if isinstance(val, str):
        v = val.strip()
        return v if v else "false"
    if isinstance(val, bool):
        return "true" if val else "false"
    return "true" if val else "false"


def _page_models(
    task: str,
    page_limit: int,
    cursor: Optional[str] = None,
    *,
    gated_filter: Optional[bool] = None,
) -> tuple[list[dict], Optional[str]]:
    params = {"pipeline_tag": task, "limit": str(page_limit)}
    if cursor:
        params["cursor"] = cursor
    if gated_filter:
        params["gated"] = "true"
    elif gated_filter is False:
        params["gated"] = "false"

    resp = requests.get(HF_API, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json()
    if not isinstance(items, list):
        items = []
    return items, _parse_next_cursor(resp)


def fetch_all_ids_by_task_gated(
    task: str, page_limit: int = 1000, hard_page_cap: int = 100
) -> Set[str]:
    gated_ids: Set[str] = set()
    cursor: Optional[str] = None
    for _ in range(hard_page_cap):
        page, next_cursor = _page_models(
            task, page_limit, cursor, gated_filter=True
        )
        for m in page:
            if m.get("private", False):
                continue
            mid = m.get("id")
            if mid:
                gated_ids.add(mid)
        if next_cursor:
            cursor = next_cursor
        elif len(page) < page_limit:
            break
        else:
            break
    return gated_ids


def fetch_all_by_task(
    task: str, page_limit: int = 1000, hard_page_cap: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch all non-private *transformers* models for a task.
    Annotate 'gated' as "manual"/"true"/"false".
    """
    all_items: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(hard_page_cap):
        page, next_cursor = _page_models(
            task, page_limit, cursor, gated_filter=None
        )
        public = [
            m
            for m in page
            if not m.get("private", False)
            and (
                m.get("library_name") == "transformers"
                or m.get("libraryName") == "transformers"
            )
        ]
        all_items.extend(public)
        if next_cursor:
            cursor = next_cursor
        elif len(page) < page_limit:
            break
        else:
            break

    gated_ids = fetch_all_ids_by_task_gated(
        task, page_limit=page_limit, hard_page_cap=hard_page_cap
    )
    for m in all_items:
        raw = m.get("gated", False)
        if isinstance(raw, str) and raw.strip():
            m["gated"] = raw  # keep e.g. "manual"
        else:
            m["gated"] = (
                "true" if (m.get("id") in gated_ids or bool(raw)) else "false"
            )
    return all_items
