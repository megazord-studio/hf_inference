"""
app/main.py

FastAPI application for HuggingFace model inference.

Endpoints:
- GET /healthz - health check endpoint
- POST /inference - inference endpoint accepting multipart form data
- GET / - model sorting and filtering UI
"""

import io
import json
import os
from typing import Any
from typing import Dict
from typing import Optional

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import ValidationError

from app.helpers import device_str
from app.routes import hf_models
from app.runners import RUNNERS

app = FastAPI(title="HF Inference API", version="0.1.0")
# Use a package-relative path for static assets so it works when installed
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in spec: {str(e)}"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid spec format: {str(e)}"
        )

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

    # Build the spec for the runner
    runner_spec = {
        "model_id": inference_spec.model_id,
        "task": task,
        "payload": inference_spec.payload.copy(),
        "files": {
            "image": image,
            "audio": audio,
            "video": video,
        },
    }

    dev = device_str()

    try:
        result = runner(runner_spec, dev)

        # Handle different result types
        if isinstance(result, dict):
            # Check if result contains file data
            if (
                "file_data" in result
                and "file_name" in result
                and "content_type" in result
            ):
                # Return file as streaming response
                return StreamingResponse(
                    io.BytesIO(result["file_data"]),
                    media_type=result["content_type"],
                    headers={
                        "Content-Disposition": f"attachment; filename={result['file_name']}"
                    },
                )
            elif "files" in result:
                # Multiple files - return first one for now (can be enhanced)
                # For simplicity, return JSON with base64 encoded files or URLs
                return JSONResponse(content=result)
            else:
                # Regular JSON response
                return JSONResponse(content=result)
        else:
            return JSONResponse(content={"result": result})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{task} inference failed",
                "reason": str(e),
            },
        )


def main() -> None:
    """Run the HF Inference API with default host and port."""
    import uvicorn

    host = os.getenv("HF_INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("HF_INFERENCE_PORT", "8000"))
    reload = os.getenv("HF_INFERENCE_RELOAD", "0") == "1"

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
