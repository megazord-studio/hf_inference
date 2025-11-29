"""Minimal ResourceManager (Phase 0).

Provides a simple memory watermark check using psutil.
Configuration is hardcoded for simplicity per project guidelines.
"""

from __future__ import annotations

import logging

log = logging.getLogger("app.resources")

try:  # optional dependency, already likely present in environment
    import psutil
except Exception:  # pragma: no cover
    psutil = None


class ResourceManager:
    def __init__(self) -> None:
        # Hardcoded watermark to avoid env-driven behavior; can be tuned in code.
        self._watermark_percent = 85.0  # % RAM usage trigger

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
