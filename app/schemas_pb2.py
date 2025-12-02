"""Auto-generated from proto/contracts.proto.

Do not edit manually; run `poe generate-contracts` instead."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class TaskOutputMetadata(BaseModel):
    task: str
    runtime_ms_model: Optional[int] = None
    resolved_model_id: Optional[str] = None
    backend: Optional[str] = None


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


class InferenceResult(BaseModel):
    task_output: Dict[str, Any]
    echo: Dict[str, Any]
    info: Dict[str, Any]
    metadata: TaskOutputMetadata


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


class StreamingTokenPayload(BaseModel):
    type: Optional[str] = None
    index: Optional[int] = None
    text: Optional[str] = None


class StreamingProgressPayload(BaseModel):
    type: Optional[str] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    percent: Optional[float] = None
    chunk_index: Optional[int] = None
    num_chunks: Optional[int] = None
    audio_base64: Optional[str] = None


class StreamingDonePayload(BaseModel):
    type: Optional[str] = None
    tokens: Optional[int] = None
    runtime_ms: Optional[int] = None
    first_token_latency_ms: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model_id: Optional[str] = None
    task: Optional[str] = None
    steps: Optional[int] = None
    image_base64: Optional[str] = None
    num_chunks: Optional[int] = None


class StreamingErrorPayload(BaseModel):
    type: Optional[str] = None
    message: Optional[str] = None
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TaskCategory(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    tasks: List[str] = Field(default_factory=list)


class TaskInfo(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    supported: Optional[bool] = None


__all__ = [
    "ErrorResponse",
    "TaskOutputMetadata",
    "ModelMeta",
    "ModelSummary",
    "InferenceResult",
    "InferenceResponsePayload",
    "InferenceErrorPayload",
    "StreamingEvent",
    "StreamingTokenPayload",
    "StreamingProgressPayload",
    "StreamingDonePayload",
    "StreamingErrorPayload",
    "TaskCategory",
    "TaskInfo",
]
