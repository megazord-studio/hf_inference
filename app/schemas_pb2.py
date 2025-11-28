"""Auto-generated from proto/contracts.proto.

Do not edit manually; run `poe generate-contracts` instead."""
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
    task_output: Dict[str, Any]
    echo: Dict[str, Any]
    info: Dict[str, Any]
    metadata: TaskOutputMetadata

class ModelMeta(BaseModel):
    id: str
    model_id: Optional[str] = None
    author: Optional[str] = None
    gated: Optional[bool] = None
    private: Optional[bool] = None
    last_modified: Optional[str] = None
    created_at: Optional[str] = None
    likes: Optional[int] = None
    trending_score: Optional[float] = None
    downloads: Optional[int] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    sha: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    card_data: Optional[Dict[str, Any]] = None
    siblings: List[Dict[str, Any]] = Field(default_factory=list)
    fallback: Optional[bool] = None

class ModelSummary(BaseModel):
    id: Optional[str] = None
    pipeline_tag: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    gated: Optional[bool] = None
    likes: Optional[int] = None
    downloads: Optional[int] = None
    card_data: Optional[Dict[str, Any]] = None

class InferenceResponsePayload(BaseModel):
    result: Optional[InferenceResult] = None
    runtime_ms: Optional[int] = None
    model_id: Optional[str] = None
    model_meta: Optional[ModelMeta] = None
    error: Optional[ErrorResponse] = None

class InferenceErrorPayload(BaseModel):
    error: ErrorResponse

class StreamingEvent(BaseModel):
    type: Optional[str] = None
    version: Optional[str] = None
    correlation_id: Optional[str] = None
    task: Optional[str] = None
    model_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

__all__ = ["ErrorResponse", "TaskOutputMetadata", "InferenceResult", "ModelMeta", "ModelSummary", "InferenceResponsePayload", "InferenceErrorPayload", "StreamingEvent"]
