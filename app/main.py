import io
import json
import logging
import os
import time
import asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.middleware import Middleware

from app.auth import SharedSecretAuthMiddleware
from app.helpers import device_str
from app.routes import hf_models
import app.routes.auth_routes as auth_routes
from app.runners import RUNNERS

logger = logging.getLogger("uvicorn.error")

middleware = [
    Middleware(SharedSecretAuthMiddleware, env_var="INFERENCE_SHARED_SECRET")
]

app = FastAPI(title="HF Inference API", version="0.1.0", middleware=middleware)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(auth_routes.router)
app.include_router(hf_models.router)


class InferenceSpec(BaseModel):
    model_id: str
    task: str
    payload: Dict[str, Any] = {}
    extra_args: Dict[str, Any] = {}  # no max_tokens anymore


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"status": "ok", "device": device_str()}


@app.post("/inference")
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
        return JSONResponse(content=result if isinstance(result, dict) else {"result": result})
    except asyncio.CancelledError:
        raise HTTPException(status_code=504, detail="Client disconnected")
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail={"error": str(e)})


def main() -> None:
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
