"""FastAPI dependency injection.

Provides dependency functions for accessing PRIME instance
and other shared resources in route handlers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002 - needed at runtime for FastAPI DI

if TYPE_CHECKING:
    from prime import PRIME
    from prime.evaluation import RAGASEvaluator


def get_prime(request: Request) -> PRIME:
    """Get PRIME instance from app state.

    Args:
        request: FastAPI request object.

    Returns:
        PRIME instance stored in app state.

    Raises:
        RuntimeError: If PRIME not initialized.
    """
    prime = getattr(request.app.state, "prime", None)
    if prime is None:
        raise RuntimeError("PRIME not initialized")
    return prime  # type: ignore[return-value]


def get_evaluator(request: Request) -> RAGASEvaluator:
    """Get RAGASEvaluator instance from app state.

    Args:
        request: FastAPI request object.

    Returns:
        RAGASEvaluator instance stored in app state.

    Raises:
        RuntimeError: If evaluator not initialized or disabled.
    """
    evaluator = getattr(request.app.state, "evaluator", None)
    if evaluator is None:
        raise RuntimeError(
            "RAGAS evaluator not initialized. "
            "Ensure OPENAI_API_KEY is set and ragas.enabled=True in config."
        )
    return evaluator  # type: ignore[return-value]
