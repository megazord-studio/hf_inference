import logging
import os
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import socket
from app.core.device import ensure_task_supported
from app.core.registry import REGISTRY, PIPELINE_TO_TASK
from app.core.runners import SUPPORTED_TASKS
from app.core.tasks import get_output_model_for_task
from fastapi.responses import StreamingResponse
import uuid
import time
import json
import base64

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

# Full model enrichment helper using model_info with expand list (now with retries & fallback)
def _fetch_full_model_meta(model_id: str) -> Optional[dict]:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    expand_fields = [
        "author","cardData","config","createdAt","lastModified","likes","trendingScore",
        "downloads","pipeline_tag","library_name","sha","tags","siblings","gated","private"
    ]
    retries = int(os.getenv("HF_META_RETRIES", "2"))
    timeout = float(os.getenv("HF_META_TIMEOUT", "10"))
    last_err: Exception | None = None
    for attempt in range(1, retries + 2):  # initial + retries
        try:
            info = api.model_info(model_id, expand=expand_fields, timeout=timeout)
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
            meta["_id"] = getattr(info, "_id", None)
            return meta
        except (HfHubHTTPError, socket.timeout, ConnectionError, OSError, Exception) as e:  # broad catch for transient issues
            last_err = e
            # Only warn on final attempt; earlier attempts debug for noise reduction
            level = log.warning if attempt == (retries + 1) else log.debug
            level(f"model_info enrichment attempt {attempt} failed for {model_id}: {e}")
    # Fallback: try lightweight list_models search for minimal data
    try:
        listed = api.list_models(search=f"id={model_id}", limit=1)
        listed_models = list(listed)
        if listed_models:
            m = listed_models[0]
            return {
                "id": model_id,
                "modelId": model_id,
                "likes": getattr(m, "likes", None),
                "downloads": getattr(m, "downloads", None),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "tags": getattr(m, "tags", None),
                "gated": getattr(m, "gated", None),
                "private": getattr(m, "private", None),
                "fallback": True,
            }
    except Exception as e2:
        log.debug(f"Fallback list_models failed for {model_id}: {e2}")
    log.warning(f"Model meta unavailable for {model_id}; proceeding without enrichment (last error: {last_err})")
    return None

# Placeholder store for future model handles / pooling / concurrency limits
# For now we simply echo request; real inference integration will plug into HF Inference Endpoints or local pipelines.

@router.post("/inference", response_model=InferenceResponse)
async def run_inference(req: InferenceRequest, include_model_meta: bool = True) -> InferenceResponse:
    """Inference endpoint dispatching to local runners (text, vision/audio)."""
    if not req.inputs:
        raise HTTPException(status_code=400, detail="inputs object cannot be empty")
    provided = {k: ("<bytes>" if isinstance(v, str) and v.startswith("data:") else v) for k, v in req.inputs.items()}
    result: Dict[str, Any] = {
        "echo": provided,
        "info": {
            "model_id": req.model_id,
            "intent_id": req.intent_id or "",
            "input_type": req.input_type,
            "received_fields": list(req.inputs.keys()),
        }
    }
    meta: dict | None = None
    if include_model_meta:
        meta = _fetch_full_model_meta(req.model_id)
        if meta is None:
            log.warning(f"Model meta unavailable for {req.model_id}; proceeding without enrichment")
    # Device suitability enforcement
    try:
        ensure_task_supported(req.task or (meta.get("pipeline_tag") if meta else None))
    except RuntimeError as e:
        result["task_output"] = {}
        result["error"] = {"message": str(e)}
        return InferenceResponse(result=result, runtime_ms=0, model_id=req.model_id, model_meta=meta)

    task = req.task
    if not task and meta and meta.get("pipeline_tag") in PIPELINE_TO_TASK:
        task = PIPELINE_TO_TASK[meta.get("pipeline_tag")]

    if task and task in SUPPORTED_TASKS:
        try:
            pred = REGISTRY.predict(task=task, model_id=req.model_id, inputs=req.inputs, options=req.options or {})
            output_model = get_output_model_for_task(task)
            raw_output = pred["output"]
            if output_model is not None:
                try:
                    parsed = output_model.model_validate(raw_output)
                    task_output = parsed.model_dump()
                except Exception:
                    task_output = raw_output
            else:
                task_output = raw_output
            # Preserve backend and other runner-level metadata if present
            if isinstance(raw_output, dict) and "backend" in raw_output:
                task_output.setdefault("backend", raw_output["backend"])
            result["task_output"] = task_output
            result["task"] = task
            result["runtime_ms_model"] = pred["runtime_ms"]
            result["resolved_model_id"] = pred.get("resolved_model_id")
            if "backend" in pred:
                result["backend"] = pred["backend"]
        except Exception as e:
            result.setdefault("task_output", {})
            result["task"] = task
            result["error"] = {"message": f"inference_failed: {e}"}
    else:
        # Task is unknown or not yet implemented in the backend runners
        result["task_output"] = {}
        if task:
            result["task"] = str(task)
            result["error"] = {"message": f"task_not_implemented: {task}"}
        else:
            result["error"] = {"message": "task_not_provided"}
    return InferenceResponse(result=result, runtime_ms=0, model_id=req.model_id, model_meta=meta)

@router.get("/models/status")
async def models_status():
    """Expose runtime status for all loaded models, including load time and errors."""
    return {"loaded": REGISTRY.list_loaded()}

@router.delete("/models/{task}/{model_id}")
async def unload_model(task: str, model_id: str):
    ok = REGISTRY.unload(task, model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model not loaded")
    return {"unloaded": f"{model_id}:{task}"}

@router.get("/inference/stream")
async def stream_inference(model_id: str, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_p: float = 1.0):
    """Server-Sent Events streaming for text generation (manual SSE implementation).

    Emits events: token, done (and optionally error). Uses a simple synchronous generation
    followed by incremental emission to keep implementation deterministic for tests.
    """
    corr_id = str(uuid.uuid4())
    task = "text-generation"
    start_time = time.time()
    try:
        pred_init = REGISTRY.predict(
            task=task,
            model_id=model_id,
            inputs={"text": prompt},
            options={"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "_stream": True},
        )
        tokens = pred_init["output"].get("tokens", [])
    except Exception as e:
        async def error_iter():
            yield f"event: error\nid: {corr_id}\ndata: {json.dumps({'message': str(e)})}\n\n"
        return StreamingResponse(error_iter(), media_type="text/event-stream")

    async def event_iter():
        first_token_latency_ms: Optional[int] = None
        # Start event (optional; aids debugging)
        yield f"event: start\nid: {corr_id}\ndata: {json.dumps({'model_id': model_id, 'token_count_planned': len(tokens)})}\n\n"
        for i, tok in enumerate(tokens):
            if first_token_latency_ms is None:
                first_token_latency_ms = int((time.time() - start_time) * 1000)
            payload = {"index": i, "text": tok}
            yield f"event: token\nid: {corr_id}\ndata: {json.dumps(payload)}\n\n"
        total_ms = int((time.time() - start_time) * 1000)
        tps = round(len(tokens) / (max(total_ms, 1) / 1000.0), 2) if tokens else 0.0
        done_payload = {
            "tokens": len(tokens),
            "runtime_ms": total_ms,
            "first_token_latency_ms": first_token_latency_ms,
            "tokens_per_second": tps,
        }
        yield f"event: done\nid: {corr_id}\ndata: {json.dumps(done_payload)}\n\n"
    return StreamingResponse(event_iter(), media_type="text/event-stream")


@router.get("/inference/stream/text-to-image")
async def stream_text_to_image(model_id: str, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 7.5):
    """SSE streaming for text-to-image diffusion with synthetic progress events.

    We run a single blocking diffusion call, then emit a sequence of `progress`
    events followed by a final `done` event containing the generated image. This
    keeps the implementation deterministic for tests without complex callbacks.
    """
    corr_id = str(uuid.uuid4())
    task = "text-to-image"
    start_time = time.time()
    try:
        pred = REGISTRY.predict(
            task=task,
            model_id=model_id,
            inputs={"text": prompt},
            options={"num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "_stream": True},
        )
        output = pred["output"] or {}
        image_b64 = output.get("image_base64")
        runtime_ms = int(output.get("runtime_ms") or (time.time() - start_time) * 1000)
        steps = int(output.get("num_inference_steps") or num_inference_steps)
    except Exception as exc:
        error_msg = str(exc)
        async def error_iter():
            payload = {"message": error_msg}
            yield f"event: error\nid: {corr_id}\ndata: {json.dumps(payload)}\n\n"
        return StreamingResponse(error_iter(), media_type="text/event-stream")

    async def event_iter():
        total_steps = max(1, steps)
        # Start event for debugging / correlation
        yield f"event: start\nid: {corr_id}\ndata: {json.dumps({'model_id': model_id, 'total_steps': total_steps})}\n\n"
        # Emit synthetic progress steps; keep it small for tests
        for i in range(total_steps):
            payload = {
                "step": i + 1,
                "total_steps": total_steps,
                "percent": round(((i + 1) / total_steps) * 100.0, 1),
            }
            yield f"event: progress\nid: {corr_id}\ndata: {json.dumps(payload)}\n\n"
        done_payload: Dict[str, Any] = {
            "model_id": model_id,
            "task": task,
            "steps": total_steps,
            "runtime_ms": runtime_ms,
            "image_base64": image_b64,
        }
        yield f"event: done\nid: {corr_id}\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@router.get("/inference/stream/tts")
async def stream_tts(model_id: str, text: str):
    """SSE streaming for text-to-speech using simple chunked audio.

    We generate a full TTS waveform via REGISTRY.predict, then split the
    base64-encoded WAV bytes into chunks and emit them as progress events.
    """
    corr_id = str(uuid.uuid4())
    task = "text-to-speech"
    start_time = time.time()
    try:
        pred = REGISTRY.predict(
            task=task,
            model_id=model_id,
            inputs={"text": text},
            options={"_stream": True},
        )
        output = pred["output"] or {}
        audio_b64 = output.get("audio_base64")
        sample_rate = output.get("sample_rate")
        num_samples = output.get("num_samples")
        if not audio_b64:
            raise RuntimeError("tts_missing_audio")
        header, data = audio_b64.split(",", 1)
        raw_bytes = base64.b64decode(data)
    except Exception as e:
        async def error_iter():
            yield f"event: error\nid: {corr_id}\ndata: {json.dumps({'message': str(e)})}\n\n"
        return StreamingResponse(error_iter(), media_type="text/event-stream")

    async def event_iter():
        chunk_size = 64 * 1024
        total_len = len(raw_bytes)
        num_chunks = max(1, (total_len + chunk_size - 1) // chunk_size)
        # Start event with basic metadata
        start_payload = {
            "model_id": model_id,
            "task": task,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "num_chunks": num_chunks,
        }
        yield f"event: start\nid: {corr_id}\ndata: {json.dumps(start_payload)}\n\n"
        for idx in range(num_chunks):
            chunk = raw_bytes[idx * chunk_size : (idx + 1) * chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            payload = {
                "chunk_index": idx,
                "num_chunks": num_chunks,
                "audio_base64": f"{header},{chunk_b64}",
            }
            yield f"event: progress\nid: {corr_id}\ndata: {json.dumps(payload)}\n\n"
        runtime_ms = int((time.time() - start_time) * 1000)
        done_payload = {
            "model_id": model_id,
            "task": task,
            "runtime_ms": runtime_ms,
            "num_chunks": num_chunks,
        }
        yield f"event: done\nid: {corr_id}\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")
