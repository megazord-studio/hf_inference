"""Centralized task taxonomy constants and output schemas.

Provides categorized sets of all tasks exposed in the frontend plus
Pydantic models describing the standard shape of runner outputs.
"""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Type

from pydantic import BaseModel
from pydantic import Field

# --- Task taxonomy (from runners) -------------------------------------------

from app.core.runners.multimodal import MULTIMODAL_TASKS as MULTIMODAL_TASKS_RUNNERS
from app.core.runners.retrieval import RETRIEVAL_TASKS as RETRIEVAL_TASKS_RUNNERS
from app.core.runners.text import TEXT_TASKS as TEXT_TASKS_RUNNERS
from app.core.runners.video_generation import VIDEO_TASKS as VIDEO_TASKS_RUNNERS
from app.core.runners.vision import VISION_TASKS
from app.core.runners.audio import AUDIO_TASKS
from app.core.runners.vision_generation import VISION_GEN_TASKS
from app.core.runners.vision_understanding import VISION_UNDERSTANDING_TASKS

# Combined for backwards compatibility
VISION_AUDIO_TASKS_RUNNERS = VISION_TASKS | AUDIO_TASKS

# Existing frontend-oriented taxonomy remains for compatibility
GENERATION_VISION_TASKS: Set[str] = {
    "text-to-image", "image-to-image", "image-super-resolution", "image-restoration",
    "image-to-3d", "text-to-3d",
}
VIDEO_TASKS: Set[str] = {"text-to-video", "image-to-video"}
AUDIO_EXTENDED_TASKS: Set[str] = {
    "audio-to-audio", "text-to-audio", "audio-text-to-text", "voice-activity-detection",
}
RETRIEVAL_TASKS: Set[str] = {"visual-document-retrieval"}
FORECASTING_TASKS: Set[str] = {"time-series-forecasting"}
MULTIMODAL_TASKS: Set[str] = {"image-text-to-text"}  # VQA / reasoning
GENERALIST_TASKS: Set[str] = {"any-to-any"}
ADVANCED_VISION_UNDERSTANDING: Set[str] = {
    "zero-shot-image-classification", "zero-shot-object-detection", "keypoint-detection",
}
TEXT_EXTENDED_TASKS: Set[str] = {"text-ranking", "sentence-similarity"}

ALL_FRONTEND_TASKS: Set[str] = set().union(
    TEXT_TASKS_RUNNERS,
    VISION_AUDIO_TASKS_RUNNERS,
    GENERATION_VISION_TASKS,
    VIDEO_TASKS,
    AUDIO_EXTENDED_TASKS,
    RETRIEVAL_TASKS,
    FORECASTING_TASKS,
    MULTIMODAL_TASKS,
    GENERALIST_TASKS,
    ADVANCED_VISION_UNDERSTANDING,
    TEXT_EXTENDED_TASKS,
)

SUPPORTED_TASKS: Set[str] = set(TEXT_TASKS_RUNNERS).union(VISION_AUDIO_TASKS_RUNNERS)
UNSUPPORTED_TASKS: Set[str] = ALL_FRONTEND_TASKS - SUPPORTED_TASKS

# --- Output schemas ---------------------------------------------------------


class TextGenerationOutput(BaseModel):
    text: str


class TextClassificationLabel(BaseModel):
    label: str
    score: float


class TextClassificationOutput(BaseModel):
    predictions: List[TextClassificationLabel]


class EmbeddingOutput(BaseModel):
    embedding: List[float]
    dim: Optional[int] = None


class ImageOutput(BaseModel):
    data_url: str = Field(..., description="Image encoded as data:image/...;base64,")
    width: Optional[int] = None
    height: Optional[int] = None


class DepthSummary(BaseModel):
    mean: float
    min: float
    max: float
    shape: List[int]
    len: int


class DetectionBox(BaseModel):
    label: str
    score: float
    box: List[float]


class SegmentationOutput(BaseModel):
    labels: Dict[str, int]
    shape: List[int]


class AudioOutput(BaseModel):
    audio_base64: str


class VideoOutput(BaseModel):
    video_base64: str


class ThreeDOutput(BaseModel):
    glb_base64: str


class VisionGenerationOutput(BaseModel):
    images: List[ImageOutput]


class VisionUnderstandingOutput(BaseModel):
    # Flexible container; runners may populate only relevant fields
    predictions: Optional[List[TextClassificationLabel]] = None
    detections: Optional[List[DetectionBox]] = None
    segmentation: Optional[SegmentationOutput] = None
    depth_summary: Optional[DepthSummary] = None
    labels: Optional[Dict[str, Any]] = None
    scores: Optional[List[float]] = None
    keypoints: Optional[Any] = None
    count: Optional[int] = None


class VisionAudioOutput(BaseModel):
    text: Optional[str] = None
    predictions: Optional[List[TextClassificationLabel]] = None
    audio: Optional[AudioOutput] = None
    audio_base64: Optional[str] = None
    segments: Optional[Any] = None


class VideoTaskOutput(BaseModel):
    video: VideoOutput


class ThreeDTaskOutput(BaseModel):
    mesh: ThreeDOutput


class MultimodalTaskOutput(BaseModel):
    text: Optional[str] = None
    answer: Optional[str] = None
    arch: Optional[str] = None


class RetrievalTaskOutput(BaseModel):
    items: List[Dict[str, Any]]


TASK_TO_OUTPUT_MODEL: Dict[str, Type[BaseModel]] = {}

for t in TEXT_TASKS_RUNNERS:
    if t == "text-generation":
        continue
    if t in {"text-classification"}:
        TASK_TO_OUTPUT_MODEL[t] = TextClassificationOutput
    elif t in {"feature-extraction", "embedding"}:
        TASK_TO_OUTPUT_MODEL[t] = EmbeddingOutput
    elif t == "summarization":
        TASK_TO_OUTPUT_MODEL[t] = TextGenerationOutput

for t in VISION_GEN_TASKS:
    TASK_TO_OUTPUT_MODEL[t] = VisionGenerationOutput

for t in VISION_UNDERSTANDING_TASKS:
    TASK_TO_OUTPUT_MODEL[t] = VisionUnderstandingOutput

for t in VISION_AUDIO_TASKS_RUNNERS:
    # Do not enforce a strict schema for vision/audio tasks; allow raw runner
    # outputs (detections, labels, depth_summary, audio_base64, segments, etc.)
    # to pass through untouched so existing contracts remain valid.
    #
    # TASK_TO_OUTPUT_MODEL[t] is intentionally left unset here.
    pass

for t in VIDEO_TASKS_RUNNERS:
    TASK_TO_OUTPUT_MODEL[t] = VideoTaskOutput

for t in MULTIMODAL_TASKS_RUNNERS:
    TASK_TO_OUTPUT_MODEL[t] = MultimodalTaskOutput

for t in RETRIEVAL_TASKS_RUNNERS:
    TASK_TO_OUTPUT_MODEL[t] = RetrievalTaskOutput


def get_output_model_for_task(task: str) -> Optional[Type[BaseModel]]:
    """Return the Pydantic model describing the output for a given task.

    If a task is not yet mapped, None is returned and the raw runner
    output is forwarded as-is.
    """
    return TASK_TO_OUTPUT_MODEL.get(task)


__all__ = [
    "GENERATION_VISION_TASKS", "VIDEO_TASKS", "AUDIO_EXTENDED_TASKS", "RETRIEVAL_TASKS",
    "FORECASTING_TASKS", "MULTIMODAL_TASKS", "GENERALIST_TASKS", "ADVANCED_VISION_UNDERSTANDING",
    "TEXT_EXTENDED_TASKS", "ALL_FRONTEND_TASKS", "SUPPORTED_TASKS", "UNSUPPORTED_TASKS",
    "TextGenerationOutput",
    "TextClassificationLabel",
    "TextClassificationOutput",
    "EmbeddingOutput",
    "ImageOutput",
    "DepthSummary",
    "DetectionBox",
    "SegmentationOutput",
    "AudioOutput",
    "VideoOutput",
    "ThreeDOutput",
    "VisionGenerationOutput",
    "VisionUnderstandingOutput",
    "VisionAudioOutput",
    "VideoTaskOutput",
    "ThreeDTaskOutput",
    "MultimodalTaskOutput",
    "RetrievalTaskOutput",
    "TASK_TO_OUTPUT_MODEL",
    "get_output_model_for_task",
]
