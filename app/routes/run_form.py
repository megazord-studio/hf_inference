from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from fastapi import APIRouter
from fastapi import File
from fastapi import Query
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from app.form_schemas import get_fields_for_task

_templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

router = APIRouter(prefix="/run-form", tags=["run-form"])
logger = logging.getLogger("uvicorn.error")


def _bad_request(code: str, reason: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        {"error": code, "reason": reason, **extra}, status_code=400
    )


def _collect_files(form: Any) -> Dict[str, Optional[UploadFile]]:
    get = form.get
    return {
        "image": get("image")
        if isinstance(get("image"), UploadFile)
        else None,
        "audio": get("audio")
        if isinstance(get("audio"), UploadFile)
        else None,
        "video": get("video")
        if isinstance(get("video"), UploadFile)
        else None,
    }


def _get_form_text(form: Any, key: str) -> str:
    v = form.get(key)
    return v if isinstance(v, str) else ""


def _build_payload(
    form: Any, task: str
) -> tuple[Dict[str, Any], Optional[JSONResponse]]:
    """Build payload by walking the known schema for the task.

    - File fields are skipped (handled separately)
    - JSON fields are parsed; on error, returns a 400 response
    - Empty JSON fields are ignored
    """
    payload: Dict[str, Any] = {}
    for fld in get_fields_for_task(task):
        name = fld.get("name")
        kind = fld.get("type")
        if not name or kind == "file":
            continue
        val = form.get(name)
        if isinstance(val, UploadFile):
            continue
        if kind == "json":
            text = (val or "").strip() if isinstance(val, str) else ""
            if not text:
                continue
            try:
                payload[name] = json.loads(text)
            except json.JSONDecodeError as e:
                return {}, _bad_request("invalid_json", str(e), field=name)
        else:
            if isinstance(val, str):
                payload[name] = val
    return payload, None


def _parse_extra_args(
    text: str,
) -> tuple[Dict[str, Any], Optional[JSONResponse]]:
    text = (text or "").strip()
    if not text:
        return {}, None
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}, _bad_request(
                "invalid_extra_args", "extra_args must be a JSON object"
            )
        return data, None
    except Exception as e:
        return {}, _bad_request("invalid_extra_args", str(e))


def _to_requests_file(
    u: Optional[UploadFile],
) -> Optional[Tuple[str, Any, str]]:
    if not u:
        return None
    try:
        u.file.seek(0)
    except Exception:
        pass
    return (
        u.filename or "upload",
        u.file,
        u.content_type or "application/octet-stream",
    )


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
    """Accept modal submission, build an inference spec, and forward to /inference."""
    form = await request.form()

    task = _get_form_text(form, "task").strip()
    model_id = _get_form_text(form, "model_id").strip()
    if not task or not model_id:
        return _bad_request(
            "bad_request", "Both 'task' and 'model_id' are required."
        )

    # Prefer bound UploadFile params; fall back to form-based collection if needed
    files = {"image": image, "audio": audio, "video": video}
    if not any(files.values()):
        files = _collect_files(form)

    payload, err = _build_payload(form, task)
    if err:
        return err

    extra_args, err = _parse_extra_args(_get_form_text(form, "extra_args"))
    if err:
        return err

    spec = {
        "model_id": model_id,
        "task": task,
        "payload": payload,
        "extra_args": extra_args,
    }

    logger.info(
        "Run form submit: task=%s model=%s payload_keys=%s files=%s",
        task,
        model_id,
        list(payload.keys()),
        ",".join([k for k, v in files.items() if v is not None]) or "none",
    )

    # Call inference endpoint directly and wrap response in HTML
    from app.routes.inference import inference as run_inference
    import time

    start = time.time()
    try:
        response = await run_inference(
            spec=json.dumps(spec),
            image=files["image"],
            audio=files["audio"],
            video=files["video"],
        )
        duration = time.time() - start

        # Check if response is JSON
        if hasattr(response, "body"):
            result_data = json.loads(response.body.decode("utf-8"))
        else:
            # Handle other response types
            result_data = {"result": "Success"}

        # Determine result type for display
        result_type = "json"
        formatted_data = json.dumps(result_data, indent=2)
        
        if isinstance(result_data, list):
            result_type = "list"
            formatted_data = [json.dumps(item, indent=2) if isinstance(item, dict) else str(item) for item in result_data]
        elif isinstance(result_data, dict):
            # Check if it's a simple text result
            if "result" in result_data and isinstance(result_data["result"], str):
                result_type = "text"
                formatted_data = result_data["result"]
            else:
                formatted_data = json.dumps(result_data, indent=2)

        return templates.TemplateResponse(
            "partials/result_display.html",
            {
                "request": request,
                "error": False,
                "result_data": formatted_data,
                "result_type": result_type,
                "task": task,
                "model_id": model_id,
                "duration": duration,
            },
        )
    except Exception as e:
        duration = time.time() - start
        error_msg = str(e)
        
        # Extract more detailed error message if available
        if hasattr(e, "detail"):
            if isinstance(e.detail, dict):
                error_msg = json.dumps(e.detail, indent=2)
            else:
                error_msg = str(e.detail)
        
        logger.exception("Inference failed in run_form")
        
        return templates.TemplateResponse(
            "partials/result_display.html",
            {
                "request": request,
                "error": True,
                "error_message": error_msg,
                "task": task,
                "model_id": model_id,
                "duration": duration,
            },
        )
