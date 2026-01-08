"""PRIME API exceptions.

Defines the exception hierarchy for PRIME API operations including
configuration errors, component failures, and API-specific errors.
"""

from __future__ import annotations


class PRIMEError(Exception):
    """Base PRIME exception.

    All PRIME-specific exceptions inherit from this class.
    Provides structured error codes for API responses.

    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error code for API responses.
    """

    def __init__(self, message: str, error_code: str) -> None:
        """Initialize PRIME exception.

        Args:
            message: Human-readable error description.
            error_code: Machine-readable error code (e.g., "PRIME_CONFIG_ERROR").
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ConfigurationError(PRIMEError):
    """Configuration validation failed.

    Raised when PRIME configuration is invalid or missing
    required values.
    """

    def __init__(self, message: str) -> None:
        """Initialize configuration error.

        Args:
            message: Description of the configuration problem.
        """
        super().__init__(message, "PRIME_CONFIG_ERROR")


class ComponentError(PRIMEError):
    """Component initialization or operation failed.

    Raised when a PRIME component (SSM, MCS, Predictor, Y-Encoder)
    fails to initialize or encounters an operational error.

    Attributes:
        component: Name of the failed component.
    """

    def __init__(self, component: str, message: str) -> None:
        """Initialize component error.

        Args:
            component: Name of the failed component.
            message: Description of the failure.
        """
        super().__init__(f"{component}: {message}", f"PRIME_{component.upper()}_ERROR")
        self.component = component


class SessionError(PRIMEError):
    """Session-related error.

    Raised for session management failures such as invalid
    session IDs or session state corruption.

    Attributes:
        session_id: The problematic session identifier.
    """

    def __init__(self, session_id: str, message: str) -> None:
        """Initialize session error.

        Args:
            session_id: The problematic session identifier.
            message: Description of the session problem.
        """
        super().__init__(f"Session {session_id}: {message}", "PRIME_SESSION_ERROR")
        self.session_id = session_id


class AuthenticationError(PRIMEError):
    """Authentication failed.

    Raised when API authentication fails due to missing
    or invalid credentials.
    """

    def __init__(self, message: str = "Authentication required") -> None:
        """Initialize authentication error.

        Args:
            message: Description of the authentication failure.
        """
        super().__init__(message, "PRIME_AUTH_ERROR")


class RateLimitError(PRIMEError):
    """Rate limit exceeded.

    Raised when a client exceeds the configured request rate limit.

    Attributes:
        retry_after: Seconds until the rate limit resets.
    """

    def __init__(self, retry_after: int) -> None:
        """Initialize rate limit error.

        Args:
            retry_after: Seconds until the rate limit resets.
        """
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after}s",
            "PRIME_RATE_LIMIT_ERROR",
        )
        self.retry_after = retry_after
