"""Runners package exports.

Centralized task -> runner registry for all supported inference tasks.
Each domain (text, vision, audio, multimodal, etc.) is organized
into its own subpackage for maintainability.
"""

from __future__ import annotations

from typing import Dict
from typing import Type

from .audio import AUDIO_TASKS
from .audio import audio_runner_for_task
from .base import BaseRunner
from .multimodal import MULTIMODAL_TASKS
from .multimodal import multimodal_runner_for_task
from .retrieval import RETRIEVAL_TASKS
from .retrieval import retrieval_runner_for_task
from .text import TEXT_TASKS
from .text import runner_for_task as text_runner_for_task
from .video_generation import VIDEO_TASKS
from .video_generation import video_runner_for_task
from .vision import VISION_TASKS
from .vision import vision_runner_for_task
from .vision_3d import VISION_3D_TASKS
from .vision_3d import vision_3d_runner_for_task
from .vision_generation import VISION_GEN_TASKS
from .vision_generation import vision_gen_runner_for_task
from .vision_understanding import VISION_UNDERSTANDING_TASKS
from .vision_understanding import vision_understanding_runner_for_task

# Centralized task -> runner class registry
TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {}

for _task in TEXT_TASKS:
    TASK_TO_RUNNER[_task] = text_runner_for_task(_task)
for _task in VISION_TASKS:
    TASK_TO_RUNNER[_task] = vision_runner_for_task(_task)
for _task in AUDIO_TASKS:
    TASK_TO_RUNNER[_task] = audio_runner_for_task(_task)
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

    Raises KeyError if the task is unknown to the backend.
    """
    return TASK_TO_RUNNER[task]


SUPPORTED_TASKS = frozenset(TASK_TO_RUNNER.keys())

# Combined task sets for backwards compatibility
VISION_AUDIO_TASKS = VISION_TASKS | AUDIO_TASKS


def runner_for_task(task: str) -> Type[BaseRunner]:
    """Alias for get_runner_cls for backwards compatibility."""
    return get_runner_cls(task)


__all__ = [
    "TEXT_TASKS",
    "text_runner_for_task",
    "VISION_TASKS",
    "vision_runner_for_task",
    "AUDIO_TASKS",
    "audio_runner_for_task",
    "VISION_AUDIO_TASKS",
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
