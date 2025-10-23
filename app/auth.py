"""Backward-compatible authentication module.

This module re-exports authentication components from the features.auth module
to maintain backward compatibility.

Deprecated: Import directly from app.features.auth modules instead.
"""

from app.features.auth.middleware import LOGIN_PASSWORD
from app.features.auth.middleware import LOGIN_USER
from app.features.auth.middleware import SESSION_COOKIE_NAME
from app.features.auth.middleware import SESSION_SIGNING_KEY
from app.features.auth.middleware import SESSION_TTL_SECONDS
from app.features.auth.middleware import SharedSecretAuthMiddleware
from app.features.auth.middleware import _sign_session
from app.features.auth.middleware import parse_session_cookie

__all__ = [
    "SharedSecretAuthMiddleware",
    "SESSION_SIGNING_KEY",
    "SESSION_COOKIE_NAME",
    "SESSION_TTL_SECONDS",
    "LOGIN_USER",
    "LOGIN_PASSWORD",
    "parse_session_cookie",
    "_sign_session",
]

