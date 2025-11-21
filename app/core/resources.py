"""Minimal ResourceManager (Phase 0).

Provides a simple memory watermark check using psutil (if available).
No side effects beyond in-process queries.
"""
from __future__ import annotations
import os
import logging

log = logging.getLogger("app.resources")

try:  # optional dependency, already likely present in environment
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

class ResourceManager:
    def __init__(self) -> None:
        self._watermark_percent = float(os.getenv("MEMORY_WATERMARK_PERCENT", "85"))  # % RAM usage trigger

    def need_eviction(self) -> bool:
        if psutil is None:
            return False
        try:
            used_percent = psutil.virtual_memory().percent
            return used_percent >= self._watermark_percent
        except Exception as e:  # pragma: no cover
            log.debug(f"Resource check failed: {e}")
            return False

RESOURCES = ResourceManager()

__all__ = ["RESOURCES", "ResourceManager"]

