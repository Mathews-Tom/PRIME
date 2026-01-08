"""API key authentication middleware.

Validates X-API-Key header for protected endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002 - needed at runtime for middleware dispatch
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.responses import Response
    from starlette.types import ASGIApp


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication.

    Validates the X-API-Key header against a set of valid keys.
    Public paths (health, docs) bypass authentication.

    Attributes:
        api_keys: Set of valid API keys.
        PUBLIC_PATHS: Paths that bypass authentication.
    """

    PUBLIC_PATHS: frozenset[str] = frozenset({
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    })

    def __init__(
        self,
        app: ASGIApp,
        api_keys: frozenset[str] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application.
            api_keys: Set of valid API keys. If None or empty, auth is disabled.
            enabled: Whether authentication is enabled.
        """
        super().__init__(app)
        self.api_keys = api_keys or frozenset()
        self.enabled = enabled and len(self.api_keys) > 0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and validate API key.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            Response from the next handler or 401 if unauthorized.
        """
        # Skip auth if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Validate API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "error_code": "MISSING_API_KEY",
                    "detail": "X-API-Key header is required",
                },
            )

        if api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "error_code": "INVALID_API_KEY",
                    "detail": "Invalid API key",
                },
            )

        return await call_next(request)
