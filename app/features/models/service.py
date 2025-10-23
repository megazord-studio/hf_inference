from __future__ import annotations

import datetime
from datetime import timedelta
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

# --- simple in-memory caches (10 min) ----------------------------------------
_CACHE_TTL = timedelta(minutes=10)
_cache_min: Dict[str, Tuple[datetime.datetime, List[Dict[str, Any]]]] = {}
_cache_full: Dict[str, Tuple[datetime.datetime, List[Dict[str, Any]]]] = {}


def get_cached_min(task: str) -> Optional[List[Dict[str, Any]]]:
    ent = _cache_min.get(task)
    if not ent:
        return None
    ts, data = ent
    return (
        data
        if (datetime.datetime.now(datetime.UTC) - ts) < _CACHE_TTL
        else None
    )


def set_cached_min(task: str, data: List[Dict[str, Any]]) -> None:
    _cache_min[task] = (datetime.datetime.now(datetime.UTC), data)


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
