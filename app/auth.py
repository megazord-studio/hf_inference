import hmac
import os
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Set

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from starlette.responses import Response
from starlette.types import ASGIApp


class SharedSecretAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforce a static Bearer token in the Authorization header.

    - Reads the secret from `secret=` or from env var (default: INFERENCE_SHARED_SECRET).
    - Supports multiple comma-separated secrets (rotation).
    - Skips configured exempt paths (exact or prefix match).
    - Returns 401 with WWW-Authenticate on failure.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret: Optional[str] = None,
        env_var: str = "INFERENCE_SHARED_SECRET",
        header_name: str = "authorization",
        scheme: str = "Bearer",
        exempt_paths: Optional[Iterable[str]] = ("/healthz", "/docs", "/redoc", "/openapi.json"),
    ) -> None:
        super().__init__(app)
        self.header_name = header_name.lower()
        self.scheme = scheme
        self.exempt: Set[str] = set(exempt_paths or ())

        if secret is not None:
            self.secrets = [secret.strip()]
        else:
            raw = os.getenv(env_var, "").strip()
            self.secrets = [s.strip() for s in raw.split(",") if s.strip()]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Exemptions (exact or prefix)
        for p in self.exempt:
            base = p.rstrip("/")
            if path == p or (base and path.startswith(base + "/")):
                return await call_next(request)

        # If no secret configured, fail-open (change to fail-closed if desired)
        if not getattr(self, "secrets", []):
            return await call_next(request)

        provided = request.headers.get(self.header_name, "")
        if not provided.startswith(self.scheme + " "):
            return self._unauthorized()

        token = provided[len(self.scheme) + 1 :].strip()

        for s in self.secrets:
            if hmac.compare_digest(token, s):
                return await call_next(request)

        return self._unauthorized()

    @staticmethod
    def _unauthorized() -> Response:
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="inference"'},
        )
