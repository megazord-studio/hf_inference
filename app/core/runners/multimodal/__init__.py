"""Multimodal runners package - Image+Text -> Text.

Exports the main runner and task set for use by the registry.
"""

from typing import Set
from typing import Type

from app.core.runners.base import BaseRunner

from .runner import ImageTextToTextRunner
from .any_to_any import AnyToAnyRunner

MULTIMODAL_TASKS: Set[str] = {"image-text-to-text", "any-to-any"}

_TASK_MAP = {"image-text-to-text": ImageTextToTextRunner, "any-to-any": AnyToAnyRunner}


def multimodal_runner_for_task(task: str) -> Type[BaseRunner]:
    """Return the runner class for a multimodal task."""
    return _TASK_MAP[task]


__all__ = [
    "MULTIMODAL_TASKS",
    "multimodal_runner_for_task",
    "ImageTextToTextRunner",
    "AnyToAnyRunner",
]
