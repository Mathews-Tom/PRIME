"""Request logging middleware.

Logs request details and timing information.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog
from fastapi import Request  # noqa: TC002 - needed at runtime for middleware dispatch
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.responses import Response
    from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging with timing.

    Logs request method, path, status code, and duration.
    Adds X-Request-ID and X-Request-Time-Ms headers to responses.

    Attributes:
        log_request_body: Whether to log request body (for debugging).
        log_response_body: Whether to log response body (for debugging).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ) -> None:
        """Initialize the logging middleware.

        Args:
            app: The ASGI application.
            log_request_body: Whether to log request bodies.
            log_response_body: Whether to log response bodies.
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and log details.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            Response with timing headers added.
        """
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Store request ID in state for access in route handlers
        request.state.request_id = request_id

        # Log request start
        log_data: dict[str, str | int | float] = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else "",
        }

        # Get client info
        if request.client:
            log_data["client_ip"] = request.client.host

        # Check for API key (log partial for security)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            log_data["api_key_prefix"] = api_key[:8] + "..."

        logger.debug("request_started", **log_data)

        # Time the request
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000

        # Log request completion
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Request-Time-Ms"] = str(round(duration_ms, 2))

        return response
