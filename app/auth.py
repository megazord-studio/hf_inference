import hmac
import os
import time
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from starlette.responses import RedirectResponse
from starlette.responses import Response
from starlette.types import ASGIApp


# Module-level session signing key so middleware and endpoints share it
SESSION_SIGNING_KEY = os.getenv("INFERENCE_SESSION_SECRET", "").strip() or os.urandom(32).hex()
SESSION_COOKIE_NAME = "session"
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12h
LOGIN_USER = (os.getenv("INFERENCE_LOGIN_USER", "admin").strip() or "admin")
LOGIN_PASSWORD = os.getenv("INFERENCE_LOGIN_PASSWORD", "").strip()


def _sign_session(user: str, ts: Optional[int] = None) -> str:
    ts = ts or int(time.time())
    sig = hmac.new(SESSION_SIGNING_KEY.encode(), f"{user}:{ts}".encode(), "sha256").hexdigest()
    return f"{user}:{ts}:{sig}"


def parse_session_cookie(val: str) -> Tuple[bool, Optional[str]]:
    try:
        user, ts_s, sig = val.split(":", 2)
        ts = int(ts_s)
    except Exception:
        return False, None
    if (time.time() - ts) > SESSION_TTL_SECONDS:
        return False, None
    want = hmac.new(SESSION_SIGNING_KEY.encode(), f"{user}:{ts}".encode(), "sha256").hexdigest()
    if hmac.compare_digest(sig, want):
        return True, user
    return False, None


class SharedSecretAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforce either:
      - Static Bearer token in the Authorization header, OR
      - Session cookie (signed) established via the /login form.

    - Reads bearer secrets from `secret=` or env var (default: INFERENCE_SHARED_SECRET, comma separated for rotation).
    - Session password from INFERENCE_LOGIN_PASSWORD (single password, username optional INFERENCE_LOGIN_USER, default 'admin').
    - Session HMAC signing secret from INFERENCE_SESSION_SECRET (random fail-open if missing).
    - Skips configured exempt paths (exact or prefix match).
    - Redirects HTML clients to /login when not authenticated, else 401.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret: Optional[str] = None,
        env_var: str = "INFERENCE_SHARED_SECRET",
        header_name: str = "authorization",
        scheme: str = "Bearer",
        exempt_paths: Optional[Iterable[str]] = (
            "/healthz",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/login",
            "/logout",
            "/static",
        ),
        session_cookie: str = SESSION_COOKIE_NAME,
        session_ttl_seconds: int = SESSION_TTL_SECONDS,
    ) -> None:
        super().__init__(app)
        self.header_name = header_name.lower()
        self.scheme = scheme
        self.exempt: Set[str] = set(exempt_paths or ())
        self.login_path = "/login"
        self.session_cookie = session_cookie
        self.session_ttl_seconds = session_ttl_seconds

        if secret is not None:
            self.secrets = [secret.strip()]
        else:
            raw = os.getenv(env_var, "").strip()
            self.secrets = [s.strip() for s in raw.split(",") if s.strip()]

        # Session password & user (optional). If no password configured -> session auth disabled (fail-open for that part)
        self.login_password = LOGIN_PASSWORD
        self.login_user = LOGIN_USER
        # Signing secret (if not set, generate ephemeral one -> sessions invalidated on restart)
        self.session_signing_key = SESSION_SIGNING_KEY

    # ----------------------------- Helpers ---------------------------------

    def _is_path_exempt(self, path: str) -> bool:
        for p in self.exempt:
            base = p.rstrip("/")
            if path == p or (base and path.startswith(base + "/")):
                return True
        return False

    def _parse_session(self, cookie_val: str) -> Tuple[bool, Optional[str]]:
        return parse_session_cookie(cookie_val)

    def _has_valid_session(self, request: Request) -> bool:
        if not self.login_password:  # session auth disabled
            return False
        cookie_val = request.cookies.get(self.session_cookie)
        if not cookie_val:
            return False
        ok, user = self._parse_session(cookie_val)
        if ok:
            request.state.user = user  # type: ignore[attr-defined]
            return True
        return False

    def _has_valid_bearer(self, request: Request) -> bool:
        if not getattr(self, "secrets", []):
            return False
        provided = request.headers.get(self.header_name, "")
        if not provided.startswith(self.scheme + " "):
            return False
        token = provided[len(self.scheme) + 1 :].strip()
        for s in self.secrets:
            if hmac.compare_digest(token, s):
                return True
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[override]
        path = request.url.path

        if self._is_path_exempt(path):
            return await call_next(request)

        # Bearer auth first (fast path)
        if self._has_valid_bearer(request):
            request.state.authenticated = True  # type: ignore[attr-defined]
            return await call_next(request)

        # Session cookie
        if self._has_valid_session(request):
            request.state.authenticated = True  # type: ignore[attr-defined]
            return await call_next(request)

        # If neither configured (no secrets & no login password) => fail-open (original behaviour when no secret)
        if not self.secrets and not self.login_password:
            return await call_next(request)

        # Unauthorized: redirect HTML clients to login, else 401
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            # Preserve original target in next parameter
            dest = f"{self.login_path}?next={path}" if path != self.login_path else self.login_path
            return RedirectResponse(dest, status_code=302)
        return self._unauthorized()

    @staticmethod
    def _unauthorized() -> Response:
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="inference"'},
        )
