"""In-process sliding-window rate limiting for expensive API requests."""

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Limit POST requests to the ingestion, query, and evaluation endpoints."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        limited_paths = {
            "/api/v1/ingest",
            "/api/v1/query",
            "/api/v1/query/stream",
            "/api/v1/evaluate",
        }
        if request.method != "POST" or request.url.path not in limited_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = monotonic()
        with self._lock:
            timestamps = self._requests[client_ip]
            while timestamps and now - timestamps[0] >= 60:
                timestamps.popleft()
            if len(timestamps) >= settings.RATE_LIMIT_RPM:
                retry_after = max(1, int(60 - (now - timestamps[0])))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please retry shortly."},
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)

        return await call_next(request)
