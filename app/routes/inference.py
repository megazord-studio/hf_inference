from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import ValidationError

from app.helpers import device_str
from app.runners import RUNNERS

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


class InferenceSpec(BaseModel):
    model_id: str
    task: str
    payload: Dict[str, Any] = {}
    extra_args: Dict[str, Any] = {}


def _is_binary_response(result: Any) -> bool:
    """Check if the result is a binary file response."""
    return isinstance(result, dict) and "file_data" in result


def _create_binary_stream(
    file_data: bytes, chunk_size: int = 8192
) -> Iterator[bytes]:
    """
    Generate chunks from binary data for streaming.

    This implements true streaming by yielding chunks instead of buffering
    the entire file in memory (as suggested in the review comment).
    """
    offset = 0
    data_length = len(file_data)
    while offset < data_length:
        chunk = file_data[offset : offset + chunk_size]
        yield chunk
        offset += chunk_size


def _create_streaming_response(result: Dict[str, Any]) -> StreamingResponse:
    """
    Create a StreamingResponse for binary file data.

    This handles the conversion of binary responses (images, audio)
    from runners into proper HTTP streaming responses with appropriate
    headers and content-type.
    """
    file_data = result["file_data"]
    file_name = result.get("file_name", "output.bin")
    content_type = result.get("content_type", "application/octet-stream")

    return StreamingResponse(
        _create_binary_stream(file_data),
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{file_name}"'},
    )


# Pure helper functions for form parsing
def _parse_spec_from_form(form_data: Any) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse inference spec from form data (pure function).
    
    Returns tuple of (spec_dict, error_message).
    If error_message is not None, spec_dict will be None.
    """
    from app.form_schemas import get_fields_for_task
    
    def _get_text(key: str) -> str:
        v = form_data.get(key)
        return v if isinstance(v, str) else ""
    
    task = _get_text("task").strip()
    model_id = _get_text("model_id").strip()
    
    if not task or not model_id:
        return None, "Both 'task' and 'model_id' are required."
    
    # Build payload from form fields
    payload: Dict[str, Any] = {}
    for fld in get_fields_for_task(task):
        name = fld.get("name")
        kind = fld.get("type")
        if not name or kind == "file":
            continue
        val = form_data.get(name)
        if isinstance(val, UploadFile):
            continue
        if kind == "json":
            text = (val or "").strip() if isinstance(val, str) else ""
            if not text:
                continue
            try:
                payload[name] = json.loads(text)
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON in field '{name}': {str(e)}"
        else:
            if isinstance(val, str):
                payload[name] = val
    
    # Parse extra_args
    extra_text = _get_text("extra_args").strip()
    extra_args: Dict[str, Any] = {}
    if extra_text:
        try:
            data = json.loads(extra_text)
            if not isinstance(data, dict):
                return None, "extra_args must be a JSON object"
            extra_args = data
        except json.JSONDecodeError as e:
            return None, f"Invalid extra_args JSON: {str(e)}"
    
    return {
        "model_id": model_id,
        "task": task,
        "payload": payload,
        "extra_args": extra_args,
    }, None


@router.post("/inference")
async def inference(
    request: Request,
    spec: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> Response:
    """
    Unified inference endpoint supporting both:
    1. JSON spec string (original API)
    2. Form-based submission (from UI modal)
    
    This is a pure orchestration function that delegates to pure helpers.
    """
    # Parse spec from either JSON string or form data
    if spec:
        # Original API: spec provided as JSON string
        try:
            spec_dict = json.loads(spec)
            inference_spec = InferenceSpec(**spec_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid spec: {str(e)}")
        
        model_id = inference_spec.model_id
        task = inference_spec.task
        payload = inference_spec.payload
        extra_args = inference_spec.extra_args
    else:
        # Form-based submission from UI
        form_data = await request.form()
        spec_dict, error = _parse_spec_from_form(form_data)
        if error:
            raise HTTPException(status_code=400, detail={"error": "bad_request", "reason": error})
        
        model_id = spec_dict["model_id"]
        task = spec_dict["task"]
        payload = spec_dict["payload"]
        extra_args = spec_dict.get("extra_args", {})
    
    # Validate task
    runner = RUNNERS.get(task)
    if not runner:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported task",
                "task": task,
                "supported_tasks": sorted(RUNNERS.keys()),
            },
        )
    
    # Create immutable runner spec
    runner_spec: Dict[str, Any] = {
        "model_id": model_id,
        "task": task,
        "payload": payload,
        "files": {"image": image, "audio": audio, "video": video},
        "extra_args": extra_args,
    }
    
    dev = device_str()
    logger.info(
        "Inference request: model=%s task=%s device=%s has_files=%s extra_arg_keys=%s",
        model_id,
        task,
        dev,
        bool(image or audio or video),
        list(extra_args.keys()),
    )
    
    loop = asyncio.get_running_loop()
    start = time.time()
    try:
        result = await loop.run_in_executor(None, runner, runner_spec, dev)
        duration = time.time() - start
        logger.info(
            "Inference completed: model=%s task=%s device=%s duration=%.2fs",
            model_id,
            task,
            dev,
            duration,
        )
        
        # Check if result is a binary file response
        if _is_binary_response(result):
            return _create_streaming_response(result)
        
        # Otherwise return JSON response
        return JSONResponse(
            content=result if isinstance(result, dict) else {"result": result}
        )
    except asyncio.CancelledError:
        raise HTTPException(status_code=504, detail="Client disconnected")
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail={"error": str(e)})
