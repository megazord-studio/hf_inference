"""Run form routes - now delegates to unified inference endpoint.

This module maintains the /run-form prefix for backward compatibility
but the actual inference logic has been merged into the main inference endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi import File
from fastapi import Query
from fastapi import Request
from fastapi import UploadFile
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
async def post_run_form(
    request: Request,
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> Response:
    """
    Delegate to main inference endpoint.
    
    The /inference endpoint now handles both spec-based and form-based submissions,
    so we simply forward the request there.
    """
    from app.routes.inference import inference
    
    # Call inference without spec - it will parse from form data
    return await inference(
        request=request,
        spec=None,  # Signal to parse from form
        image=image,
        audio=audio,
        video=video,
    )
