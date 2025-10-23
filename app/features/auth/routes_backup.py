"""Backward-compatible auth routes module.

Deprecated: Import directly from app.features.auth.routes instead.
"""

from app.features.auth.routes import router

__all__ = ["router"]
