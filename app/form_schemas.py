"""Backward-compatible form schemas module.

This module re-exports form schema components from the features.ui module
to maintain backward compatibility.

Deprecated: Import directly from app.features.ui.form_schemas instead.
"""

from app.features.ui.form_schemas import TASK_FORM_SPECS
from app.features.ui.form_schemas import UIField
from app.features.ui.form_schemas import get_fields_for_task

__all__ = ["UIField", "get_fields_for_task", "TASK_FORM_SPECS"]
