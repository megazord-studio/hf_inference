from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi import Query
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from app.form_schemas import get_fields_for_task

_templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

router = APIRouter(prefix="/run-form", tags=["run-form"])


@router.get("")
async def get_run_form(
    request: Request,
    task: Optional[str] = Query(None, description="Pipeline task key"),
    model_id: Optional[str] = Query(None, description="Selected model id"),
) -> Response:
    """Render the run form partial (HTMX target) for a given task/model."""
    fields = get_fields_for_task(task)
    return templates.TemplateResponse(
        "partials/run_form.html",
        {
            "request": request,
            "task": task or "",
            "model_id": model_id or "",
            "fields": fields,
        },
    )


@router.post("")
async def post_run_form() -> JSONResponse:
    """Placeholder endpoint until submission is wired to /inference."""
    return JSONResponse(
        {
            "error": "not_implemented",
            "reason": "Submit is disabled; wiring to /inference will be added later.",
        },
        status_code=501,
    )
