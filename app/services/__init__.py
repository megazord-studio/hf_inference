"""Backward-compatible services module.

This module re-exports service components from the features modules
to maintain backward compatibility.

Deprecated: Import directly from app.features.models.service instead.
"""

from app.features.models.service import fetch_all_by_task
from app.features.models.service import fetch_all_ids_by_task_gated
from app.features.models.service import gated_to_str
from app.features.models.service import get_cached_min
from app.features.models.service import set_cached_min

__all__ = [
    "fetch_all_by_task",
    "fetch_all_ids_by_task_gated",
    "gated_to_str",
    "get_cached_min",
    "set_cached_min",
]
