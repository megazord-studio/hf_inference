#!/usr/bin/env python3
"""
app/main.py

FastAPI application for HuggingFace model inference.

Endpoints:
- GET /healthz - health check endpoint
- POST /inference - inference endpoint accepting multipart form data
"""

import os
import sys
import json
import io
from typing import Any, Dict, Optional, Union

# --- ensure the project root is importable when running as a file path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
import uvicorn

from app.helpers import device_str
from app.runners import RUNNERS

app = FastAPI(title="HF Inference API", version="0.1.0")


class InferenceSpec(BaseModel):
    model_id: str
    task: str
    payload: Dict[str, Any] = {}


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok", "device": device_str()}


@app.post("/inference")
async def inference(
    spec: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
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
        raise HTTPException(status_code=400, detail=f"Invalid JSON in spec: {str(e)}")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid spec format: {str(e)}")
    
    task = inference_spec.task
    runner = RUNNERS.get(task)
    
    if not runner:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported task",
                "task": task,
                "supported_tasks": sorted(RUNNERS.keys()),
            }
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
        }
    }
    
    dev = device_str()
    
    try:
        result = runner(runner_spec, dev)
        
        # Handle different result types
        if isinstance(result, dict):
            # Check if result contains file data
            if "file_data" in result and "file_name" in result and "content_type" in result:
                # Return file as streaming response
                return StreamingResponse(
                    io.BytesIO(result["file_data"]),
                    media_type=result["content_type"],
                    headers={"Content-Disposition": f"attachment; filename={result['file_name']}"}
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
            }
        )


def main():
    """Run the FastAPI application with uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
