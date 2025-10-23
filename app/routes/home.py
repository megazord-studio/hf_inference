from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter
from fastapi import Query
from fastapi import Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from app.runners import RUNNERS

_templates_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "templates"
)
templates = Jinja2Templates(directory=_templates_dir)

router = APIRouter(tags=["home"])


@router.get("/")
def list_models_table(
    request: Request,
    task: Optional[str] = Query(
        None, description="Implemented pipeline tag, e.g. 'image-text-to-text'"
    ),
) -> Response:
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
