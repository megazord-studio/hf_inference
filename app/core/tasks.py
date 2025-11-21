"""Centralized task taxonomy constants (Phase A).

Provides categorized sets of all tasks exposed in the frontend. Distinguishes
currently supported vs unsupported to enable graceful fallback responses.
"""
from __future__ import annotations
from typing import Set

# Existing supported task sets are imported from runners modules
from app.core.runners.text import TEXT_TASKS
from app.core.runners.vision_audio import VISION_AUDIO_TASKS

# --- Category sets (frontend coverage) ---
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
    TEXT_TASKS,
    VISION_AUDIO_TASKS,
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

SUPPORTED_TASKS: Set[str] = set(TEXT_TASKS).union(VISION_AUDIO_TASKS)
UNSUPPORTED_TASKS: Set[str] = ALL_FRONTEND_TASKS - SUPPORTED_TASKS

__all__ = [
    "GENERATION_VISION_TASKS", "VIDEO_TASKS", "AUDIO_EXTENDED_TASKS", "RETRIEVAL_TASKS",
    "FORECASTING_TASKS", "MULTIMODAL_TASKS", "GENERALIST_TASKS", "ADVANCED_VISION_UNDERSTANDING",
    "TEXT_EXTENDED_TASKS", "ALL_FRONTEND_TASKS", "SUPPORTED_TASKS", "UNSUPPORTED_TASKS",
]
