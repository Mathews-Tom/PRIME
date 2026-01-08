"""FastAPI application factory.

Creates and configures the PRIME REST API server.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prime import PRIME, PRIMEConfig, PRIMEError
from prime.api.middleware import (
    APIKeyMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)
from prime.api.models import ErrorResponse
from prime.api.routes import (
    clusters_router,
    config_router,
    diagnostics_router,
    memory_router,
    process_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass
class MiddlewareConfig:
    """Configuration for API middleware.

    Attributes:
        enable_cors: Whether to enable CORS middleware.
        cors_origins: Allowed CORS origins. Use ["*"] for all.
        enable_rate_limit: Whether to enable rate limiting.
        requests_per_minute: Rate limit requests per minute.
        rate_limit_window: Rate limit window in seconds.
        enable_auth: Whether to enable API key authentication.
        api_keys: Set of valid API keys.
        enable_logging: Whether to enable request logging.
    """

    enable_cors: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_rate_limit: bool = True
    requests_per_minute: int = 100
    rate_limit_window: int = 60
    enable_auth: bool = False
    api_keys: frozenset[str] = field(default_factory=frozenset)
    enable_logging: bool = True

    @classmethod
    def for_testing(cls) -> MiddlewareConfig:
        """Create config suitable for testing.

        Returns:
            MiddlewareConfig with auth and rate limiting disabled.
        """
        return cls(
            enable_cors=True,
            enable_rate_limit=False,
            enable_auth=False,
            enable_logging=False,
        )

    @classmethod
    def for_production(
        cls,
        api_keys: frozenset[str],
        cors_origins: list[str] | None = None,
    ) -> MiddlewareConfig:
        """Create config suitable for production.

        Args:
            api_keys: Required API keys for authentication.
            cors_origins: Allowed CORS origins. Defaults to none.

        Returns:
            MiddlewareConfig with auth enabled.
        """
        return cls(
            enable_cors=cors_origins is not None,
            cors_origins=cors_origins or [],
            enable_rate_limit=True,
            enable_auth=True,
            api_keys=api_keys,
            enable_logging=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for app startup/shutdown.

    Initializes PRIME and optional RAGASEvaluator on startup,
    cleans up on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None after startup, cleanup happens after yield.
    """
    import os

    import structlog

    logger = structlog.get_logger(__name__)

    # Startup
    config: PRIMEConfig = getattr(app.state, "config", PRIMEConfig.for_testing())
    app.state.prime = PRIME(config)

    # Initialize RAGAS evaluator if enabled and API key is available
    if config.ragas.enabled and os.getenv("OPENAI_API_KEY"):
        try:
            from prime.evaluation import RAGASEvaluator

            app.state.evaluator = RAGASEvaluator(config.ragas)
            logger.info("ragas_evaluator_initialized", model=config.ragas.llm_model)
        except Exception as e:
            logger.warning("ragas_evaluator_failed", error=str(e))
            app.state.evaluator = None
    else:
        app.state.evaluator = None
        if config.ragas.enabled:
            logger.info("ragas_evaluator_skipped", reason="OPENAI_API_KEY not set")

    yield

    # Shutdown - no explicit cleanup needed for current implementation


def create_app(
    config: PRIMEConfig | None = None,
    middleware_config: MiddlewareConfig | None = None,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Optional PRIMEConfig. If None, uses testing config.
        middleware_config: Optional MiddlewareConfig. If None, uses testing config.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="PRIME API",
        description="Predictive Retrieval with Intelligent Memory Embeddings",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store config for lifespan handler
    if config is not None:
        app.state.config = config

    # Use testing middleware config by default
    mw_config = middleware_config or MiddlewareConfig.for_testing()

    # Register middleware (order matters - last added is first to process)
    _register_middleware(app, mw_config)

    # Register exception handlers
    _register_exception_handlers(app)

    # Include routers with API version prefix
    app.include_router(process_router, prefix="/api/v1")
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(diagnostics_router, prefix="/api/v1")
    app.include_router(clusters_router, prefix="/api/v1")
    app.include_router(config_router, prefix="/api/v1")

    return app


def _register_middleware(app: FastAPI, config: MiddlewareConfig) -> None:
    """Register middleware on the application.

    Middleware is added in reverse order of processing:
    - API Key Auth (first to process, last added)
    - Rate Limiting
    - Logging
    - CORS (last to process, first added)

    Args:
        app: FastAPI application.
        config: Middleware configuration.
    """
    # CORS middleware (last to process incoming, first to process outgoing)
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-Request-ID",
                "X-Request-Time-Ms",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
        )

    # Logging middleware
    if config.enable_logging:
        app.add_middleware(LoggingMiddleware)

    # Rate limiting middleware
    if config.enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=config.requests_per_minute,
            window_seconds=config.rate_limit_window,
            enabled=True,
        )

    # API key authentication middleware (first to process)
    if config.enable_auth:
        app.add_middleware(
            APIKeyMiddleware,
            api_keys=config.api_keys,
            enabled=True,
        )


def _register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers for the app.

    Args:
        app: FastAPI application to register handlers on.
    """

    @app.exception_handler(PRIMEError)
    async def prime_error_handler(
        request: Request,  # noqa: ARG001
        exc: PRIMEError,
    ) -> JSONResponse:
        """Handle PRIME-specific errors.

        Args:
            request: FastAPI request.
            exc: PRIMEError exception.

        Returns:
            JSON error response.
        """
        request_id = str(uuid.uuid4())

        error_response = ErrorResponse(
            error=exc.__class__.__name__,
            error_code=f"PRIME_{exc.__class__.__name__.upper()}",
            detail=str(exc),
            request_id=request_id,
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )

    @app.exception_handler(ValueError)
    async def validation_error_handler(
        request: Request,  # noqa: ARG001
        exc: ValueError,
    ) -> JSONResponse:
        """Handle validation errors.

        Args:
            request: FastAPI request.
            exc: ValueError exception.

        Returns:
            JSON error response with 400 status.
        """
        request_id = str(uuid.uuid4())

        error_response = ErrorResponse(
            error="ValidationError",
            error_code="VALIDATION_ERROR",
            detail=str(exc),
            request_id=request_id,
        )

        return JSONResponse(
            status_code=400,
            content=error_response.model_dump(),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(
        request: Request,  # noqa: ARG001
        exc: RuntimeError,
    ) -> JSONResponse:
        """Handle runtime errors.

        Args:
            request: FastAPI request.
            exc: RuntimeError exception.

        Returns:
            JSON error response.
        """
        request_id = str(uuid.uuid4())

        error_response = ErrorResponse(
            error="RuntimeError",
            error_code="RUNTIME_ERROR",
            detail=str(exc),
            request_id=request_id,
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )


# Default app instance for uvicorn
app = create_app()
