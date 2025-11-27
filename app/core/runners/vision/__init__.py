"""Vision runners package - image classification, segmentation, etc.

Exports runner classes and task set for vision-related inference.
"""
from typing import Set
from typing import Type

from app.core.runners.base import BaseRunner

from .captioning import ImageCaptioningRunner
from .classification import ImageClassificationRunner
from .depth import DepthEstimationRunner
from .detection import ObjectDetectionRunner
from .segmentation import ImageSegmentationRunner

VISION_TASKS: Set[str] = {
    "image-classification",
    "image-captioning",
    "object-detection",
    "image-segmentation",
    "depth-estimation",
}

_TASK_TO_RUNNER = {
    "image-classification": ImageClassificationRunner,
    "image-captioning": ImageCaptioningRunner,
    "object-detection": ObjectDetectionRunner,
    "image-segmentation": ImageSegmentationRunner,
    "depth-estimation": DepthEstimationRunner,
}


def vision_runner_for_task(task: str) -> Type[BaseRunner]:
    """Return the runner class for a vision task."""
    return _TASK_TO_RUNNER[task]


__all__ = [
    "VISION_TASKS",
    "vision_runner_for_task",
    "ImageClassificationRunner",
    "ImageCaptioningRunner",
    "ObjectDetectionRunner",
    "ImageSegmentationRunner",
    "DepthEstimationRunner",
]
