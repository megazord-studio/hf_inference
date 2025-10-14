from __future__ import annotations

import html
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from fastapi import APIRouter
from fastapi import Query
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

from app.runners import RUNNERS
from app.services.hf_models_service import fetch_all_by_task
from app.services.hf_models_service import gated_to_str
from app.services.hf_models_service import get_cached_min
from app.services.hf_models_service import set_cached_min
from app.views.hf_models_table import render_models_table

router = APIRouter(tags=["models"])

# ------------------------------ JSON endpoint -------------------------------


@router.get("/models", response_class=JSONResponse)
def list_models_minimal(
    task: Optional[str] = Query(
        None, description="Implemented pipeline tag, e.g. 'image-text-to-text'"
    ),
    limit: int = Query(
        1000,
        ge=1,
        le=1000,
        description="Per-page fetch size for internal pagination",
    ),
) -> JSONResponse:
    """
    Minimal JSON for ALL non-private transformers models of a task (10 min cache):
      id, likes, trendingScore, downloads, gated
    """
    if not task:
        return JSONResponse(
            {"available_tasks": sorted(RUNNERS.keys())}, status_code=200
        )

    if task not in RUNNERS:
        return JSONResponse(
            {
                "error": f"unsupported task '{task}'",
                "supported": sorted(RUNNERS.keys()),
            },
            status_code=400,
        )

    cached = get_cached_min(task)
    if cached is not None:
        return JSONResponse(cached)

    try:
        models = fetch_all_by_task(task, page_limit=limit, hard_page_cap=200)
    except Exception as e:
        return JSONResponse(
            {"error": "hf_api_failed", "reason": str(e)}, status_code=502
        )

    minimal: List[Dict[str, Any]] = [
        {
            "id": m.get("id"),
            "likes": m.get("likes", 0),
            "trendingScore": m.get("trendingScore", 0),
            "downloads": m.get("downloads", 0),
            "gated": gated_to_str(m.get("gated", "false")),
        }
        for m in models
    ]
    set_cached_min(task, minimal)
    return JSONResponse(minimal)


# ------------------------------ HTML endpoint -------------------------------


@router.get("/", response_class=HTMLResponse)
def list_models_table(
    task: Optional[str] = Query(
        None, description="Implemented pipeline tag, e.g. 'image-text-to-text'"
    ),
) -> HTMLResponse:
    """
    Virtualized table + Web Worker (client-side filtering/sorting on full dataset).
    """
    if not task:
        return HTMLResponse(render_models_table(None, []))

    if task not in RUNNERS:
        return HTMLResponse(
            f"<pre>Unsupported task: {html.escape(task)}</pre>",
            status_code=400,
        )

    # Do not inline massive JSON; the page will fetch /models?task=...
    return HTMLResponse(render_models_table(task, []))
