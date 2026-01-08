"""PRIME REST API module.

Provides FastAPI application and endpoints for the PRIME system.

Example:
    >>> from prime.api import create_app
    >>> from prime import PRIMEConfig
    >>> app = create_app(PRIMEConfig.for_testing())

Running with uvicorn:
    $ uvicorn prime.api.app:app --reload
"""

from __future__ import annotations

from prime.api.app import MiddlewareConfig, create_app
from prime.api.dependencies import get_evaluator, get_prime
from prime.api.middleware import (
    APIKeyMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)
from prime.api.models import (
    BatchEvalRequest,
    BatchEvalResponse,
    ClusterInfoResponse,
    ClusterListResponse,
    ComponentStatusResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    DiagnosticsResponse,
    ErrorResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    MemoryResult,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
    ProcessRequest,
    ProcessResponse,
)

__all__ = [
    "APIKeyMiddleware",
    "BatchEvalRequest",
    "BatchEvalResponse",
    "ClusterInfoResponse",
    "ClusterListResponse",
    "ComponentStatusResponse",
    "ConfigUpdateRequest",
    "ConfigUpdateResponse",
    "DiagnosticsResponse",
    "ErrorResponse",
    "EvalRequest",
    "EvalResponse",
    "HealthResponse",
    "LoggingMiddleware",
    "MemoryResult",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemoryWriteRequest",
    "MemoryWriteResponse",
    "MiddlewareConfig",
    "ProcessRequest",
    "ProcessResponse",
    "RateLimitMiddleware",
    "create_app",
    "get_evaluator",
    "get_prime",
]
