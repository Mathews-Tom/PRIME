"""Rate limiting middleware.

Implements sliding window rate limiting with proper headers.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002 - needed at runtime for middleware dispatch
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.responses import Response
    from starlette.types import ASGIApp


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting using sliding window algorithm.

    Limits requests per client using IP address or API key as identifier.
    Adds rate limit headers to responses and returns 429 when exceeded.

    Attributes:
        requests_per_minute: Maximum requests allowed per window.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 100,
        window_seconds: int = 60,
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            app: The ASGI application.
            requests_per_minute: Maximum requests per window.
            window_seconds: Window size in seconds.
            enabled: Whether rate limiting is enabled.
        """
        super().__init__(app)
        self.rpm = requests_per_minute
        self.window = window_seconds
        self.enabled = enabled
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request.

        Uses API key if present, otherwise falls back to IP address.

        Args:
            request: The incoming request.

        Returns:
            Client identifier string.
        """
        # Prefer API key for identification
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"

        # Fall back to IP address
        # Check X-Forwarded-For for proxied requests
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain
            return f"ip:{forwarded.split(',')[0].strip()}"

        # Use client host directly
        client = request.client
        if client:
            return f"ip:{client.host}"

        return "ip:unknown"

    def _cleanup_window(self, client_id: str, now: float) -> None:
        """Remove expired entries from the client's request window.

        Args:
            client_id: The client identifier.
            now: Current timestamp.
        """
        window = self._requests[client_id]
        cutoff = now - self.window
        while window and window[0] < cutoff:
            window.popleft()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and apply rate limiting.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            Response with rate limit headers, or 429 if limit exceeded.
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)

        client_id = self._get_client_id(request)
        now = time.time()

        # Clean up expired entries
        self._cleanup_window(client_id, now)

        window = self._requests[client_id]
        remaining = self.rpm - len(window)

        # Check if rate limit exceeded
        if remaining <= 0:
            retry_after = int(window[0] + self.window - now) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": "TooManyRequests",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "detail": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(window[0] + self.window)),
                },
            )

        # Record this request
        window.append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.window))

        return response

    def reset(self) -> None:
        """Reset all rate limit counters.

        Useful for testing.
        """
        self._requests.clear()
