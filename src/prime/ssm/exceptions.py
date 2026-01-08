"""Exception hierarchy for Semantic State Monitor.

Defines custom exceptions for SSM operations following the
explicit error handling strategy (fail fast, no silent failures).
"""

from __future__ import annotations


class SSMError(Exception):
    """Base exception for SSM errors.

    All SSM-specific exceptions inherit from this class,
    enabling targeted exception handling.

    Example:
        >>> try:
        ...     ssm.update("")
        ... except SSMError as e:
        ...     logger.error(f"SSM operation failed: {e}")
    """


class EncodingError(SSMError):
    """Raised when text encoding fails.

    This can occur due to:
    - Empty or whitespace-only input
    - Encoder model failure
    - Dimension mismatch between encoder output and configuration
    """


class ConfigurationError(SSMError):
    """Raised when SSM configuration is invalid.

    This can occur due to:
    - Invalid threshold values
    - Invalid window size
    - Incompatible parameter combinations
    """


class InsufficientDataError(SSMError):
    """Raised when window has insufficient data for variance calculation.

    This error indicates the sliding window buffer doesn't have
    enough embeddings to compute meaningful variance statistics.

    Note:
        In practice, the SSM returns variance=0.0 with CONTINUE action
        instead of raising this exception, as insufficient data during
        warm-up is expected behavior, not an error.
    """
