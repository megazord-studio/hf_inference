import logging
import os
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import HfApi
from app.core.device import ensure_task_supported
from app.core.registry import REGISTRY, PIPELINE_TO_TASK
from app.core.runners.text import TEXT_TASKS

router = APIRouter(prefix="/api", tags=["inference"])
log = logging.getLogger("app.inference")

# Request/response schemas (align with frontend expectations)
class InferenceRequest(BaseModel):
    model_id: str = Field(..., description="Hugging Face model identifier")
    intent_id: Optional[str] = Field(None, description="Frontend intent id (semantic grouping)")
    input_type: str = Field(..., description="Primary input modality selected (text, image, audio, video, document)")
    inputs: Dict[str, Any] = Field(..., description="Payload of inputs (text, image_base64, audio_base64, etc.)")
    task: Optional[str] = Field(None, description="Explicit HF task/pipeline tag (optional) for future runner dispatch")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task-specific generation / inference options")

class InferenceResponse(BaseModel):
    result: Any
    runtime_ms: Optional[int] = None
    model_id: Optional[str] = None
    model_meta: Optional[dict] = None

class CurlExampleResponse(BaseModel):
    command: str

# Full model enrichment helper using model_info with expand list
def _fetch_full_model_meta(model_id: str) -> Optional[dict]:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    expand_fields = [
        "author","cardData","config","createdAt","lastModified","likes","trendingScore",
        "downloads","pipeline_tag","library_name","sha","tags","siblings","gated","private"
    ]
    try:
        info = api.model_info(model_id, expand=expand_fields)  # richer direct fetch
        meta = {
            "id": getattr(info, "modelId", None) or model_id,
            "modelId": getattr(info, "modelId", None),
            "author": getattr(info, "author", None),
            "gated": getattr(info, "gated", None),
            "private": getattr(info, "private", None),
            "lastModified": getattr(info, "lastModified", None),
            "createdAt": getattr(info, "createdAt", None),
            "likes": getattr(info, "likes", None),
            "trendingScore": getattr(info, "trendingScore", None),
            "downloads": getattr(info, "downloads", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "sha": getattr(info, "sha", None),
            "tags": getattr(info, "tags", None),
            "config": getattr(info, "config", None),
            "cardData": getattr(info, "cardData", None),
            "siblings": [ {"rfilename": s.rfilename} for s in getattr(info, "siblings", []) ] if getattr(info, "siblings", None) else None,
        }
        meta["_id"] = getattr(info, "_id", None)  # may be absent
        return meta
    except Exception as e:
        log.warning(f"model_info enrichment failed for {model_id}: {e}")
        return None

# Placeholder store for future model handles / pooling / concurrency limits
# For now we simply echo request; real inference integration will plug into HF Inference Endpoints or local pipelines.

@router.post("/inference", response_model=InferenceResponse)
async def run_inference(req: InferenceRequest, include_model_meta: bool = True) -> InferenceResponse:
    """Stub inference endpoint with optional model metadata enrichment."""
    if not req.inputs:
        raise HTTPException(status_code=400, detail="inputs object cannot be empty")
    provided = {k: ("<bytes>" if isinstance(v, str) and v.startswith("data:") else v) for k, v in req.inputs.items()}
    result = {
        "echo": provided,
        "info": {
            "model_id": req.model_id,
            "intent_id": req.intent_id,
            "input_type": req.input_type,
            "received_fields": list(req.inputs.keys()),
        }
    }
    meta: dict | None = None
    if include_model_meta:
        meta = _fetch_full_model_meta(req.model_id)
        if meta is None:
            log.warning(f"Model meta unavailable for {req.model_id}; proceeding without enrichment")
    # New: enforce device availability for GPU-required tasks
    try:
        ensure_task_supported(req.task or meta.get("pipeline_tag") if meta else None)
    except RuntimeError as e:
        return InferenceResponse(result={"error": "device_unavailable", "detail": str(e), "suggestion": "Try smaller model or enable GPU/MPS."}, runtime_ms=0, model_id=req.model_id, model_meta=meta)

    # Phase 0 dispatch for text tasks with pipeline fallback
    task = req.task
    if not task and meta and meta.get("pipeline_tag") in PIPELINE_TO_TASK:
        task = PIPELINE_TO_TASK[meta.get("pipeline_tag")]
    if task in TEXT_TASKS:
        try:
            pred = REGISTRY.predict(task=task, model_id=req.model_id, inputs=req.inputs, options=req.options or {})
            result["task_output"] = pred["output"]
            result["task"] = task
            result["runtime_ms_model"] = pred["runtime_ms"]
            result["resolved_model_id"] = pred.get("resolved_model_id")
        except Exception as e:
            result["error"] = {"message": f"inference_failed: {e}"}
    return InferenceResponse(result=result, runtime_ms=0, model_id=req.model_id, model_meta=meta)

@router.post("/curl-example", response_model=CurlExampleResponse)
async def curl_example(req: InferenceRequest) -> CurlExampleResponse:
    """Return a curl command example for the given request.

    This mirrors the payload the frontend would send for /api/inference.
    """
    import json
    body = json.dumps(req.model_dump(), separators=(",", ":"))
    command = (
        "curl -s -X POST 'http://localhost:8000/api/inference' "
        "-H 'Content-Type: application/json' "
        f"-d '{body}'"
    )
    return CurlExampleResponse(command=command)

@router.get("/models/status")
async def models_status():
    return {"loaded": REGISTRY.list_loaded()}

@router.delete("/models/{task}/{model_id}")
async def unload_model(task: str, model_id: str):
    ok = REGISTRY.unload(task, model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model not loaded")
    return {"unloaded": f"{model_id}:{task}"}
