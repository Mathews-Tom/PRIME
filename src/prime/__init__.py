"""PRIME: Predictive Retrieval with Intelligent Memory Embeddings.

PRIME is a next-generation RAG system that predicts what context is needed
rather than reactively searching. Based on Meta FAIR's VL-JEPA architecture.

Public API:
    PRIME: Main orchestration class integrating all components
    PRIMEConfig: Main configuration class
    APIConfig: API server configuration
    RAGASConfig: RAGAS evaluation configuration
    ActionState: SSM action states (CONTINUE, PREPARE, RETRIEVE, RETRIEVE_CONSOLIDATE)
    MemoryReadResult: Memory retrieval result
    MemoryWriteResult: Memory write result
    PRIMEResponse: Response from process_turn
    PRIMEDiagnostics: System diagnostics
    ComponentStatus: Individual component health
    PRIMEError: Base exception
    ConfigurationError: Configuration validation failed
    ComponentError: Component operation failed
    SessionError: Session management error
    AuthenticationError: API authentication failed
    RateLimitError: Rate limit exceeded
"""

from __future__ import annotations

from prime.config import APIConfig, PRIMEConfig, RAGASConfig
from prime.exceptions import (
    AuthenticationError,
    ComponentError,
    ConfigurationError,
    PRIMEError,
    RateLimitError,
    SessionError,
)
from prime.prime import PRIME
from prime.types import (
    ActionState,
    ComponentStatus,
    MemoryReadResult,
    MemoryWriteResult,
    PRIMEDiagnostics,
    PRIMEResponse,
)

__version__ = "0.1.0"

__all__ = [
    "PRIME",
    "APIConfig",
    "ActionState",
    "AuthenticationError",
    "ComponentError",
    "ComponentStatus",
    "ConfigurationError",
    "MemoryReadResult",
    "MemoryWriteResult",
    "PRIMEConfig",
    "PRIMEDiagnostics",
    "PRIMEError",
    "PRIMEResponse",
    "RAGASConfig",
    "RateLimitError",
    "SessionError",
]
