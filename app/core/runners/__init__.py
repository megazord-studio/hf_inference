"""Runners package exports (multimodal removed)."""
from .text import TEXT_TASKS, runner_for_task as text_runner_for_task
try:
    from .vision_audio import VISION_AUDIO_TASKS, runner_for_task as vision_audio_runner_for_task
except Exception:  # pragma: no cover
    VISION_AUDIO_TASKS = set()  # type: ignore
    def vision_audio_runner_for_task(task: str):  # type: ignore
        raise ValueError(f"Vision/audio runner unavailable for task {task}")
from .vision_generation import VISION_GEN_TASKS, vision_gen_runner_for_task

__all__ = [
    "TEXT_TASKS",
    "text_runner_for_task",
    "VISION_AUDIO_TASKS",
    "vision_audio_runner_for_task",
    "VISION_GEN_TASKS",
    "vision_gen_runner_for_task",
]
