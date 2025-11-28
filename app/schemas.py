from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class TaskOutputMetadata(BaseModel):
    task: str
    runtime_ms_model: Optional[int] = None
    resolved_model_id: Optional[str] = None
    backend: Optional[str] = None

class InferenceResult(BaseModel):
    task_output: Any
    metadata: TaskOutputMetadata
    echo: Dict[str, Any]
    info: Dict[str, Any]

class ModelMeta(BaseModel):
    id: str
    modelId: Optional[str] = None
    author: Optional[str] = None
    gated: Optional[bool] = None
    private: Optional[bool] = None
    lastModified: Optional[str] = None
    createdAt: Optional[str] = None
    likes: Optional[int] = None
    trendingScore: Optional[float] = None
    downloads: Optional[int] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    sha: Optional[str] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    cardData: Optional[Dict[str, Any]] = None
    siblings: Optional[List[Dict[str, Any]]] = None
    fallback: Optional[bool] = None

class InferenceResponsePayload(BaseModel):
    result: InferenceResult
    runtime_ms: Optional[int] = None
    model_id: Optional[str] = None
    model_meta: Optional[ModelMeta] = None

class InferenceErrorPayload(BaseModel):
    error: ErrorResponse
    runtime_ms: Optional[int] = None
    model_id: Optional[str] = None

class StreamingEvent(BaseModel):
    type: str
    data: Dict[str, Any]

