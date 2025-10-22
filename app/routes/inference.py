from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from typing import Dict
from typing import Optional

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response
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
    extra_args: Dict[str, Any] = {}  # no max_tokens anymore


@router.post("/inference")
async def inference(
    spec: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> Response:
    try:
        spec_dict = json.loads(spec)
        inference_spec = InferenceSpec(**spec_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid spec: {str(e)}")

    task = inference_spec.task
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

    runner_spec = {
        "model_id": inference_spec.model_id,
        "task": task,
        "payload": inference_spec.payload.copy(),
        "files": {"image": image, "audio": audio, "video": video},
        "extra_args": dict(inference_spec.extra_args or {}),
    }

    dev = device_str()
    logger.info(
        "Inference request: model=%s task=%s device=%s has_files=%s extra_arg_keys=%s",
        inference_spec.model_id,
        task,
        dev,
        bool(image or audio or video),
        list((inference_spec.extra_args or {}).keys()),
    )

    loop = asyncio.get_running_loop()
    start = time.time()
    try:
        result = await loop.run_in_executor(None, runner, runner_spec, dev)
        duration = time.time() - start
        logger.info(
            "Inference completed: model=%s task=%s device=%s duration=%.2fs",
            inference_spec.model_id,
            task,
            dev,
            duration,
        )
        return JSONResponse(
            content=result if isinstance(result, dict) else {"result": result}
        )
    except asyncio.CancelledError:
        raise HTTPException(status_code=504, detail="Client disconnected")
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail={"error": str(e)})
