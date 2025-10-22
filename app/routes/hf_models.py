from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import requests

from app.runners import RUNNERS
from app.task_schemas import get_schema
from app.services.hf_models_service import (
    fetch_all_by_task,
    gated_to_str,
    get_cached_min,
    set_cached_min,
)

_templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=_templates_dir)

router = APIRouter(tags=["models"])


@router.get("/run", response_class=JSONResponse)
def get_run_schema(
    task: str = Query(..., description="Pipeline tag for the inference modal"),
) -> JSONResponse:
    """Return the UI schema for a task (defines form fields for the inference modal)."""
    if task not in RUNNERS:
        return JSONResponse(
            {"error": f"unsupported task '{task}'", "supported": sorted(RUNNERS.keys())},
            status_code=400,
        )
    
    return JSONResponse({"task": task, "schema": get_schema(task)})


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
        return JSONResponse({"available_tasks": sorted(RUNNERS.keys())}, status_code=200)

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
    except requests.HTTPError as e:
        # Return stale cache if available when rate limited
        stale = get_cached_min(task, allow_stale=True)
        status = e.response.status_code if e.response is not None else 502
        headers = {"X-HF-Error": str(e)}
        if stale is not None:
            headers["X-HF-Cache"] = "stale"
            return JSONResponse(stale, headers=headers, status_code=200)
        # No stale cache - return error
        hint = None
        if status == 429:
            hint = "Hugging Face rate limit hit. Wait a bit and retry or set HF_TOKEN."
        payload = {"error": "hf_api_failed", "reason": str(e), **({"hint": hint} if hint else {})}
        return JSONResponse(payload, status_code=status, headers=headers)
    except Exception as e:
        return JSONResponse({"error": "hf_api_failed", "reason": str(e)}, status_code=502)

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


@router.get("/")
def list_models_table(
    request: Request,
    task: Optional[str] = Query(
        None, description="Implemented pipeline tag, e.g. 'image-text-to-text'"
    ),
):
    """
    Virtualized table + Web Worker (client-side filtering/sorting on full dataset).
    """
    tasks = sorted(RUNNERS.keys())
    invalid_task = task is not None and task not in RUNNERS

    return templates.TemplateResponse(
        "models.html",
        {
            "request": request,
            "tasks": tasks,
            "task": task,
            "invalid_task": invalid_task,
        },
        status_code=400 if invalid_task else 200,
    )
