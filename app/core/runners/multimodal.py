"""Multimodal runners removed for Phase 2 focus.
This module intentionally left minimal; previous LLaVA/TinyLLaVA implementation deprecated.
"""

MULTIMODAL_TASKS = set()

def runner_for_task(task: str):
    raise RuntimeError("Multimodal tasks disabled in this build")

__all__ = ["MULTIMODAL_TASKS", "runner_for_task"]
