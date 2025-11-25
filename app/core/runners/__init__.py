"""Runners package exports (multimodal plus specialized runners).

Phase A: introduce a centralized task -> runner registry so that the
ModelRegistry and routers can avoid manual if/elif chains. This keeps
all task wiring in one place next to the concrete runner classes.
"""
from __future__ import annotations
from typing import Dict, Type

from .base import BaseRunner
from .text import TEXT_TASKS, runner_for_task as text_runner_for_task
from .vision_audio import VISION_AUDIO_TASKS, runner_for_task as vision_audio_runner_for_task
from .vision_generation import VISION_GEN_TASKS, vision_gen_runner_for_task
from .vision_understanding import (
    VISION_UNDERSTANDING_TASKS,
    vision_understanding_runner_for_task,
)
from .vision_3d import VISION_3D_TASKS, vision_3d_runner_for_task
from .multimodal import MULTIMODAL_TASKS, multimodal_runner_for_task
from .video_generation import VIDEO_TASKS, video_runner_for_task
from .retrieval import RETRIEVAL_TASKS, retrieval_runner_for_task

# Centralized task -> runner class registry (Phase A)
TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {}

for _task in TEXT_TASKS:
    TASK_TO_RUNNER[_task] = text_runner_for_task(_task)
for _task in VISION_AUDIO_TASKS:
    TASK_TO_RUNNER[_task] = vision_audio_runner_for_task(_task)
for _task in VISION_GEN_TASKS:
    TASK_TO_RUNNER[_task] = vision_gen_runner_for_task(_task)
for _task in VISION_UNDERSTANDING_TASKS:
    TASK_TO_RUNNER[_task] = vision_understanding_runner_for_task(_task)
for _task in VISION_3D_TASKS:
    TASK_TO_RUNNER[_task] = vision_3d_runner_for_task(_task)
for _task in MULTIMODAL_TASKS:
    TASK_TO_RUNNER[_task] = multimodal_runner_for_task(_task)
for _task in VIDEO_TASKS:
    TASK_TO_RUNNER[_task] = video_runner_for_task(_task)
for _task in RETRIEVAL_TASKS:
    TASK_TO_RUNNER[_task] = retrieval_runner_for_task(_task)


def get_runner_cls(task: str) -> Type[BaseRunner]:
    """Return the concrete runner class for a given normalized task.

    Raises KeyError if the task is unknown to the backend. The API layer
    is responsible for turning this into a user-facing HTTP error.
    """
    return TASK_TO_RUNNER[task]


SUPPORTED_TASKS = frozenset(TASK_TO_RUNNER.keys())

__all__ = [
    "TEXT_TASKS",
    "text_runner_for_task",
    "VISION_AUDIO_TASKS",
    "vision_audio_runner_for_task",
    "VISION_GEN_TASKS",
    "vision_gen_runner_for_task",
    "VISION_UNDERSTANDING_TASKS",
    "vision_understanding_runner_for_task",
    "VISION_3D_TASKS",
    "vision_3d_runner_for_task",
    "MULTIMODAL_TASKS",
    "multimodal_runner_for_task",
    "VIDEO_TASKS",
    "video_runner_for_task",
    "RETRIEVAL_TASKS",
    "retrieval_runner_for_task",
    "TASK_TO_RUNNER",
    "get_runner_cls",
    "SUPPORTED_TASKS",
]
