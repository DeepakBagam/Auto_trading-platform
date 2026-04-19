"""
Prometheus middleware for FastAPI.
Automatically tracks API request metrics.
"""
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from observability.metrics import api_request_duration_seconds, api_requests_total


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track API metrics with Prometheus."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path

        # Skip metrics endpoint itself
        if path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Record metrics
        api_requests_total.labels(method=method, endpoint=path, status=response.status_code).inc()
        api_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)

        return response
