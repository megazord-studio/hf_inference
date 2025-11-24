"""Runners package exports (multimodal plus specialized runners)."""
from .text import TEXT_TASKS, runner_for_task as text_runner_for_task
from .vision_audio import VISION_AUDIO_TASKS, runner_for_task as vision_audio_runner_for_task
from .vision_generation import VISION_GEN_TASKS, vision_gen_runner_for_task
from .vision_understanding import VISION_UNDERSTANDING_TASKS, vision_understanding_runner_for_task
from .vision_3d import VISION_3D_TASKS, vision_3d_runner_for_task
from .multimodal import MULTIMODAL_TASKS, multimodal_runner_for_task
from .video_generation import VIDEO_TASKS, video_runner_for_task

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
]
