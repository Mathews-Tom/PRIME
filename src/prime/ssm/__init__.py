"""Semantic State Monitor (SSM) module for PRIME.

Provides intelligent retrieval triggering by monitoring semantic
trajectories and detecting significant boundary crossings.

Public API:
    SemanticStateMonitor: Main monitor class for variance-based detection
    SSMConfig: Configuration schema for SSM parameters
    ActionState: Enum of SSM output states (CONTINUE, PREPARE, RETRIEVE, etc.)
    SemanticStateUpdate: Result type from SSM update operations
    SSMError: Base exception for SSM operations
    EncodingError: Raised when text encoding fails
    ConfigurationError: Raised when configuration is invalid

Example:
    >>> from prime.encoder import YEncoder, YEncoderConfig
    >>> from prime.ssm import SemanticStateMonitor, SSMConfig
    >>>
    >>> encoder = YEncoder(YEncoderConfig())
    >>> ssm = SemanticStateMonitor(encoder, SSMConfig())
    >>>
    >>> result = ssm.update("Tell me about Python")
    >>> if result.action == ActionState.RETRIEVE:
    ...     # Trigger retrieval operation
    ...     pass
"""

from __future__ import annotations

from prime.ssm.exceptions import (
    ConfigurationError,
    EncodingError,
    InsufficientDataError,
    SSMError,
)
from prime.ssm.ssm_config import SSMConfig
from prime.ssm.ssm_types import ActionState, SemanticStateUpdate

__all__ = [
    "ActionState",
    "ConfigurationError",
    "EncodingError",
    "InsufficientDataError",
    "SemanticStateMonitor",
    "SemanticStateUpdate",
    "SSMConfig",
    "SSMError",
]


def __getattr__(name: str) -> type:
    """Lazy import SemanticStateMonitor to avoid circular imports."""
    if name == "SemanticStateMonitor":
        from prime.ssm.ssm import SemanticStateMonitor

        return SemanticStateMonitor
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
