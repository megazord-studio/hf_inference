"""
app/main.py

FastAPI application for HuggingFace model inference.

Endpoints:
- GET /healthz    - health check endpoint
- POST /inference - inference endpoint accepting multipart form data
- GET /           - model sorting and filtering UI
- GET/POST /login - user login
- GET /logout     - user logout
"""

import io
import json
import logging
import os
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
    """Specification for an inference request."""

    model_id: str
    task: str
    payload: Dict[str, Any] = {}


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "device": device_str()}


@app.post("/inference")
async def inference(
    spec: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> Response:
    """
    Inference endpoint accepting multipart form data.

    - spec: JSON string with model_id, task, and payload
    - image: optional image file
    - audio: optional audio file
    - video: optional video file
    """
    try:
        spec_dict = json.loads(spec)
        inference_spec = InferenceSpec(**spec_dict)
    except json.JSONDecodeError as e:
        logger.info("Invalid JSON in spec: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON in spec: {str(e)}")
    except ValidationError as e:
        logger.info("Invalid spec format: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid spec format: {str(e)}")

    task = inference_spec.task
    runner = RUNNERS.get(task)

    if not runner:
        logger.info("Unsupported task requested: %s", task)
        raise HTTPException(
            status_code=400,
            detail={"error": "Unsupported task", "task": task, "supported_tasks": sorted(RUNNERS.keys())},
        )

    runner_spec = {
        "model_id": inference_spec.model_id,
        "task": task,
        "payload": inference_spec.payload.copy(),
        "files": {"image": image, "audio": audio, "video": video},
    }

    dev = device_str()

    # Log a concise request summary (do not log file contents)
    try:
        has_files = bool(image or audio or video)
        payload_keys = list(inference_spec.payload.keys()) if isinstance(inference_spec.payload, dict) else []
        logger.info(
            "Inference request: model=%s task=%s device=%s has_files=%s payload_keys=%s",
            inference_spec.model_id,
            task,
            dev,
            has_files,
            payload_keys,
        )
    except Exception:
        # Protect logging from unexpected payload types
        logger.info("Inference request: model=%s task=%s device=%s", inference_spec.model_id, task, dev)

    try:
        result = runner(runner_spec, dev)

        if isinstance(result, dict):
            if ("file_data" in result and "file_name" in result and "content_type" in result):
                return StreamingResponse(
                    io.BytesIO(result["file_data"]),
                    media_type=result["content_type"],
                    headers={"Content-Disposition": f"attachment; filename={result['file_name']}"},
                )
            elif "files" in result:
                return JSONResponse(content=result)
            else:
                return JSONResponse(content=result)
        else:
            return JSONResponse(content={"result": result})

    except Exception as e:
        logger.exception("Inference failed for model=%s task=%s device=%s", inference_spec.model_id, task, dev)
        raise HTTPException(status_code=500, detail={"error": f"{task} inference failed", "reason": str(e)})


def main() -> None:
    """Run the HF Inference API with default host and port."""
    import uvicorn

    host = os.getenv("HF_INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("HF_INFERENCE_PORT", "8000"))
    reload = os.getenv("HF_INFERENCE_RELOAD", "0") == "1"

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
