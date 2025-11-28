import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import socket
from app.core.device import ensure_task_supported
from app.core.registry import REGISTRY, PIPELINE_TO_TASK
from app.core.runners import SUPPORTED_TASKS
from app.core.tasks import get_output_model_for_task
from fastapi.responses import StreamingResponse, JSONResponse
import uuid, time, json, base64
from app.config import HF_META_RETRIES, HF_META_TIMEOUT_SECONDS
from app.routers.models import ModelSummary
from app.schemas_pb2 import (
    ErrorResponse,
    InferenceErrorPayload,
    InferenceResponsePayload,
    InferenceResult,
    ModelMeta,
    TaskOutputMetadata,
)
from datetime import datetime

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
    error: Optional[ErrorResponse] = None

# Full model enrichment helper using model_info with expand list (now with retries & fallback)
def _fetch_full_model_meta(model_id: str) -> Optional[dict]:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    retries = HF_META_RETRIES
    timeout = HF_META_TIMEOUT_SECONDS
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            info = api.model_info(model_id, timeout=timeout)  # omit expand to satisfy type constraints
            meta = _build_model_meta(info, model_id)
            return meta
        except (HfHubHTTPError, socket.timeout, ConnectionError, OSError, Exception) as e:
            last_err = e
            level = log.warning if attempt == (retries + 1) else log.debug
            level(f"model_info enrichment attempt {attempt} failed for {model_id}: {e}")
    try:
        listed = api.list_models(search=f"id={model_id}", limit=1)
        listed_models = list(listed)
        if listed_models:
            m = listed_models[0]
            return {
                "id": model_id,
                "model_id": model_id,
                "likes": getattr(m, "likes", None),
                "downloads": getattr(m, "downloads", None),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "tags": getattr(m, "tags", None),
                "gated": getattr(m, "gated", None),
                "private": getattr(m, "private", None),
                "fallback": True,
                "last_modified": None,
                "created_at": None,
                "trending_score": None,
                "library_name": None,
                "sha": None,
                "config": None,
                "card_data": None,
                "siblings": None,
            }
    except Exception as e2:
        raise RuntimeError(f"Fallback list_models failed for {model_id}: {e2}")
    log.warning(f"Model meta unavailable for {model_id}; proceeding without enrichment (last error: {last_err})")
    return None

def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):  # ensure instance, not class
        try:
            dc = asdict(value)
            return _to_serializable(dc)
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _to_serializable(value.to_dict())
    obj_dict = getattr(value, "__dict__", None)
    if obj_dict:
        serializable = {k: v for k, v in obj_dict.items() if not k.startswith("_") and not callable(v)}
        if serializable:
            return _to_serializable(serializable)
    return value

def _normalize_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        truthy = {"true", "1", "yes", "y", "gated", "manual", "auto", "enabled"}
        falsy = {"false", "0", "no", "n", "none", "null", "disabled"}
        if lowered in truthy:
            return True
        if lowered in falsy:
            return False
    return None

def _build_model_meta(info: Any, model_id: str) -> Dict[str, Any]:
    meta = {
        "id": getattr(info, "modelId", None) or model_id,
        "model_id": getattr(info, "modelId", None),
        "author": getattr(info, "author", None),
        "gated": _normalize_optional_bool(getattr(info, "gated", None)),
        "private": _normalize_optional_bool(getattr(info, "private", None)),
        "last_modified": _to_serializable(getattr(info, "lastModified", None)),
        "created_at": _to_serializable(getattr(info, "createdAt", None)),
        "likes": getattr(info, "likes", None),
        "trending_score": getattr(info, "trendingScore", None),
        "downloads": getattr(info, "downloads", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "sha": getattr(info, "sha", None),
        "tags": _to_serializable(getattr(info, "tags", None)),
        "config": _to_serializable(getattr(info, "config", None)),
        "card_data": _to_serializable(getattr(info, "cardData", None)),
        "siblings": _to_serializable(getattr(info, "siblings", None)),
    }
    return meta

def _required_fields_for_task(task: str) -> List[str]:
    mapping = {
        "text-generation": ["text"],
        "text-to-image": ["text"],
        "image-classification": ["image_base64"],
        "image-to-text": ["image_base64"],
        "object-detection": ["image_base64"],
        "image-segmentation": ["image_base64"],
        "depth-estimation": ["image_base64"],
        "image-to-image": ["image_base64"],
        "automatic-speech-recognition": ["audio_base64"],
        "text-to-speech": ["text"],
    }
    return mapping.get(task, [])

# Placeholder store for future model handles / pooling / concurrency limits
# For now we simply echo request; real inference integration will plug into HF Inference Endpoints or local pipelines.

@router.post("/inference", response_model=Union[InferenceResponsePayload, InferenceErrorPayload])
async def run_inference(req: InferenceRequest, include_model_meta: bool = True) -> JSONResponse:
    if not req.inputs:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, code="invalid_request", message="inputs object cannot be empty")
    task = req.task
    if not task and req.inputs.get("_task"):
        task = str(req.inputs.get("_task"))
    meta_raw = _fetch_full_model_meta(req.model_id) if include_model_meta else None
    meta = ModelMeta(**meta_raw) if meta_raw else None
    if task is None and meta and meta.pipeline_tag in PIPELINE_TO_TASK:
        task = PIPELINE_TO_TASK[meta.pipeline_tag]
    if not task:
        return _error_response(status.HTTP_400_BAD_REQUEST, code="missing_task", message="task must be provided or resolvable from model metadata")
    try:
        ensure_task_supported(task)
    except RuntimeError as exc:
        return _error_response(status.HTTP_400_BAD_REQUEST, code="device_unsupported", message=str(exc))
    required_fields = _required_fields_for_task(task)
    missing = [field for field in required_fields if field not in req.inputs or not req.inputs[field]]
    if missing:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, code="missing_inputs", message="Required inputs missing", details={"missing": missing})
    if task not in SUPPORTED_TASKS:
        return _error_response(status.HTTP_501_NOT_IMPLEMENTED, code="task_not_supported", message="Task is not supported by backend", details={"task": task})
    try:
        pred = REGISTRY.predict(task=task, model_id=req.model_id, inputs=req.inputs, options=req.options or {})
    except RuntimeError as exc:
        return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, code="inference_failed", message=str(exc), details={"task": task, "model_id": req.model_id})
    output_model = get_output_model_for_task(task)
    raw_output = pred["output"]
    task_output: Any = raw_output
    if output_model is not None:
        try:
            parsed = output_model.model_validate(raw_output)
            task_output = parsed.model_dump()
        except Exception:
            task_output = raw_output
    metadata = TaskOutputMetadata(task=task, runtime_ms_model=pred["runtime_ms"], resolved_model_id=pred.get("resolved_model_id"), backend=pred.get("backend"))
    inference_result = InferenceResult(
        task_output=task_output,
        metadata=metadata,
        echo={k: ("<bytes>" if isinstance(v, str) and v.startswith("data:") else v) for k, v in req.inputs.items()},
        info={"model_id": req.model_id, "intent_id": req.intent_id or "", "input_type": req.input_type, "received_fields": list(req.inputs.keys())},
    )
    payload = InferenceResponsePayload(result=inference_result, runtime_ms=pred.get("runtime_ms"), model_id=req.model_id, model_meta=meta)
    return JSONResponse(content=payload.model_dump())

@router.get("/models/status")
async def models_status() -> Dict[str, Any]:
    """Expose runtime status for all loaded models, including load time and errors."""
    return {"loaded": REGISTRY.list_loaded()}

@router.delete("/models/{task}/{model_id}")
async def unload_model(task: str, model_id: str) -> Dict[str, str]:
    ok = REGISTRY.unload(task, model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model not loaded")
    return {"unloaded": f"{model_id}:{task}"}

@router.get("/inference/stream")
async def stream_inference(model_id: str, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_p: float = 1.0) -> StreamingResponse:
    """Server-Sent Events streaming for text generation.

    SSE Event Contract (version 1.0):
    ---------------------------------
    All events include: event type, correlation id, and JSON data payload.

    Events emitted:
      - start: Initial metadata
          data: {model_id: string, token_count_planned: int}
      - token: Individual generated token
          data: {type: "token", index: int, text: string}
      - done: Generation complete with metrics
          data: {type: "done", tokens: int, runtime_ms: int,
                 first_token_latency_ms: int, tokens_per_second: float}
      - error: Error occurred
          data: {message: string}

    Example:
        event: start
        id: <uuid>
        data: {"model_id": "gpt2", "token_count_planned": 10}

        event: token
        id: <uuid>
        data: {"type": "token", "index": 0, "text": "Hello"}

        event: done
        id: <uuid>
        data: {"type": "done", "tokens": 10, "runtime_ms": 150}
    """
    corr_id = str(uuid.uuid4())
    task = "text-generation"
    start_time = time.time()
    try:
        pred_init = REGISTRY.predict(task=task, model_id=model_id, inputs={"text": prompt}, options={"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "_stream": True})
        tokens = pred_init["output"].get("tokens", [])
    except Exception as e:
        async def error_iter() -> AsyncGenerator[str, None]:
            yield f"event: error\nid: {corr_id}\ndata: {json.dumps({'message': repr(e)})}\n\n"
        return StreamingResponse(error_iter(), media_type="text/event-stream")
    async def event_iter() -> AsyncGenerator[str, None]:
        first_token_latency_ms: Optional[int] = None
        # Start event (optional; aids debugging)
        yield f"event: start\nid: {corr_id}\ndata: {json.dumps({'model_id': model_id, 'token_count_planned': len(tokens)})}\n\n"
        for i, tok in enumerate(tokens):
            if first_token_latency_ms is None:
                first_token_latency_ms = int((time.time() - start_time) * 1000)
            payload_tok = {"type": "token", "index": i, "text": tok}
            yield f"event: token\nid: {corr_id}\ndata: {json.dumps(payload_tok)}\n\n"
        total_ms = int((time.time() - start_time) * 1000)
        tps = round(len(tokens) / (max(total_ms, 1) / 1000.0), 2) if tokens else 0.0
        done_payload = {"type": "done", "tokens": len(tokens), "runtime_ms": total_ms, "first_token_latency_ms": first_token_latency_ms, "tokens_per_second": tps}
        yield f"event: done\nid: {corr_id}\ndata: {json.dumps(done_payload)}\n\n"
    return StreamingResponse(event_iter(), media_type="text/event-stream")

@router.get("/inference/stream/text-to-image")
async def stream_text_to_image(model_id: str, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 7.5) -> StreamingResponse:
    """SSE streaming for text-to-image diffusion with progress events.

    SSE Event Contract (version 1.0):
    ---------------------------------
    Events emitted:
      - start: Initial metadata
          data: {model_id: string, total_steps: int}
      - progress: Diffusion step progress
          data: {step: int, total_steps: int, percent: float}
      - done: Generation complete with image
          data: {model_id: string, task: string, steps: int,
                 runtime_ms: int, image_base64: string}
      - error: Error occurred
          data: {message: string}
    """
    corr_id = str(uuid.uuid4())
    task = "text-to-image"
    start_time = time.time()
    try:
        pred = REGISTRY.predict(task=task, model_id=model_id, inputs={"text": prompt}, options={"num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "_stream": True})
        output = pred["output"] or {}
        image_b64 = output.get("image_base64")
        runtime_ms = int(output.get("runtime_ms") or (time.time() - start_time) * 1000)
        steps = int(output.get("num_inference_steps") or num_inference_steps)
    except Exception as exc:
        error_msg = str(exc)
        async def error_iter() -> AsyncGenerator[str, None]:
            payload_err = {"message": error_msg}
            yield f"event: error\nid: {corr_id}\ndata: {json.dumps(payload_err)}\n\n"
        return StreamingResponse(error_iter(), media_type="text/event-stream")
    async def event_iter() -> AsyncGenerator[str, None]:
        total_steps = max(1, steps)
        # Start event for debugging / correlation
        yield f"event: start\nid: {corr_id}\ndata: {json.dumps({'model_id': model_id, 'total_steps': total_steps})}\n\n"
        # Emit synthetic progress steps; keep it small for tests
        for i in range(total_steps):
            payload_prog = {"step": i + 1, "total_steps": total_steps, "percent": round(((i + 1) / total_steps) * 100.0, 1)}
            yield f"event: progress\nid: {corr_id}\ndata: {json.dumps(payload_prog)}\n\n"
        done_payload: Dict[str, Any] = {"model_id": model_id, "task": task, "steps": total_steps, "runtime_ms": runtime_ms, "image_base64": image_b64}
        yield f"event: done\nid: {corr_id}\ndata: {json.dumps(done_payload)}\n\n"
    return StreamingResponse(event_iter(), media_type="text/event-stream")

@router.get("/inference/stream/tts")
async def stream_tts(model_id: str, text: str) -> StreamingResponse:
    """SSE streaming for text-to-speech with chunked audio delivery.

    SSE Event Contract (version 1.0):
    ---------------------------------
    Events emitted:
      - start: Initial metadata
          data: {model_id: string, task: string, sample_rate: int,
                 num_samples: int, num_chunks: int}
      - progress: Audio chunk
          data: {chunk_index: int, num_chunks: int, audio_base64: string}
      - done: Generation complete
          data: {model_id: string, task: string, runtime_ms: int,
                 num_chunks: int}
      - error: Error occurred
          data: {message: string}
    """
    corr_id = str(uuid.uuid4())
    task = "text-to-speech"
    start_time = time.time()
    try:
        pred = REGISTRY.predict(task=task, model_id=model_id, inputs={"text": text}, options={"_stream": True})
        output = pred["output"] or {}
        audio_b64 = output.get("audio_base64")
        sample_rate = output.get("sample_rate")
        num_samples = output.get("num_samples")
        if not audio_b64:
            raise RuntimeError("tts_missing_audio")
        header, data = audio_b64.split(",", 1)
        raw_bytes = base64.b64decode(data)
    except Exception as e:
        async def error_iter() -> AsyncGenerator[str, None]:
            yield f"event: error\nid: {corr_id}\ndata: {json.dumps({'message': repr(e)})}\n\n"
        return StreamingResponse(error_iter(), media_type="text/event-stream")
    async def event_iter() -> AsyncGenerator[str, None]:
        chunk_size = 64 * 1024
        total_len = len(raw_bytes)
        num_chunks = max(1, (total_len + chunk_size - 1) // chunk_size)
        # Start event with basic metadata
        start_payload = {"model_id": model_id, "task": task, "sample_rate": sample_rate, "num_samples": num_samples, "num_chunks": num_chunks}
        yield f"event: start\nid: {corr_id}\ndata: {json.dumps(start_payload)}\n\n"
        for idx in range(num_chunks):
            chunk = raw_bytes[idx * chunk_size : (idx + 1) * chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            payload_chunk = {"chunk_index": idx, "num_chunks": num_chunks, "audio_base64": f"{header},{chunk_b64}"}
            yield f"event: progress\nid: {corr_id}\ndata: {json.dumps(payload_chunk)}\n\n"
        runtime_ms = int((time.time() - start_time) * 1000)
        done_payload = {"model_id": model_id, "task": task, "runtime_ms": runtime_ms, "num_chunks": num_chunks}
        yield f"event: done\nid: {corr_id}\ndata: {json.dumps(done_payload)}\n\n"
    return StreamingResponse(event_iter(), media_type="text/event-stream")

def _fetch_models(limit: int) -> List[ModelSummary]:
    api = HfApi()
    results: List[ModelSummary] = []
    try:
        iterator = api.list_models(sort="downloads", direction=-1, limit=limit)
        for info in iterator:
            mid = getattr(info, "modelId", None)
            if not isinstance(mid, str):
                continue
            try:
                m = ModelSummary(
                    id=mid,
                    pipeline_tag=getattr(info, "pipeline_tag", None),
                    tags=getattr(info, "tags", None),
                    gated=getattr(info, "gated", None),
                    likes=getattr(info, "likes", None),
                    downloads=getattr(info, "downloads", None),
                    cardData=getattr(info, "cardData", None),
                )
            except Exception:
                continue
            results.append(m)
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to list models: {e}")
    return results

def _error_response(status_code: int, *, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    payload = InferenceErrorPayload(error=ErrorResponse(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=payload.model_dump())
