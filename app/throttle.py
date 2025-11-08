import asyncio
import time
from typing import Iterable, List, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

try:  # optional torch
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class ThrottleMiddleware(BaseHTTPMiddleware):
    """Simple concurrency + optional VRAM throttling for selected path prefixes.

    Defaults (no env vars):
      - Path prefixes: ['/inference']
      - Max concurrent: 1 (serializes requests)
      - Max VRAM wait: 120 seconds
      - Sleep interval while waiting: 0.25 seconds
      - VRAM threshold: disabled by default (min_free_mb=0)

    You can override via constructor params if needed.
    """

    def __init__(
        self,
        app,
        *,
        path_prefixes: Optional[Iterable[str]] = None,
        max_concurrent: int = 5,
        min_free_mb: int = 0,
        max_wait_sec: float = 120.0,
        sleep_sec: float = 0.25,
    ) -> None:
        super().__init__(app)
        self.prefixes: List[str] = list(path_prefixes or ["/inference"])  # default
        self.max_concurrent = max_concurrent
        self.min_free_mb = min_free_mb
        self.max_wait_sec = max_wait_sec
        self.sleep_sec = sleep_sec
        self._sem = asyncio.Semaphore(self.max_concurrent)

    def _matches(self, path: str) -> bool:
        for p in self.prefixes:
            b = p.rstrip("/")
            if path == p or (b and path.startswith(b + "/")):
                return True
        return False

    def _has_enough_vram(self) -> bool:
        if self.min_free_mb <= 0:
            return True
        if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():  # pragma: no cover
            return True
        free_bytes, _total_bytes = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
        return (free_bytes / (1024 ** 2)) >= self.min_free_mb

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if not self._matches(path):  # pass-through
            return await call_next(request)

        queue_start = time.time()
        await self._sem.acquire()
        queue_wait = time.time() - queue_start
        mem_wait = 0.0
        try:
            if not self._has_enough_vram():
                mem_start = time.time()
                while not self._has_enough_vram():
                    if time.time() - mem_start > self.max_wait_sec:
                        return Response(
                            content="GPU busy / insufficient free VRAM",
                            status_code=503,
                            headers={
                                "X-Queue-Wait": f"{queue_wait:.3f}",
                                "X-Mem-Wait": f"{(time.time() - mem_start):.3f}",
                                "X-Concurrency-Limit": str(self.max_concurrent),
                            },
                        )
                    await asyncio.sleep(self.sleep_sec)
                mem_wait = time.time() - mem_start

            response = await call_next(request)
            response.headers.setdefault("X-Queue-Wait", f"{queue_wait:.3f}")
            response.headers.setdefault("X-Mem-Wait", f"{mem_wait:.3f}")
            response.headers.setdefault("X-Concurrency-Limit", str(self.max_concurrent))
            return response
        finally:
            self._sem.release()
