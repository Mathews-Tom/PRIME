"""API middleware module.

Provides authentication, rate limiting, and logging middleware.

Example:
    >>> from prime.api.middleware import APIKeyMiddleware, RateLimitMiddleware
    >>> app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
    >>> app.add_middleware(APIKeyMiddleware, api_keys=frozenset({"key1"}))
"""

from __future__ import annotations

from prime.api.middleware.auth import APIKeyMiddleware
from prime.api.middleware.logging import LoggingMiddleware
from prime.api.middleware.rate_limit import RateLimitMiddleware

__all__ = [
    "APIKeyMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
]
