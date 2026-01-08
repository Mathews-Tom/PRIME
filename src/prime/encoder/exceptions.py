"""Exception hierarchy for the encoder module.

Defines specific exception types for encoder operations following
the fail-fast principle with explicit error handling.
"""

from __future__ import annotations


class EncoderError(Exception):
    """Base exception for all encoder-related errors.

    All encoder exceptions inherit from this class, enabling
    catch-all handling when needed while preserving specificity.
    """


class ModelLoadError(EncoderError):
    """Raised when encoder model fails to load.

    Causes include:
    - Model not found at specified path or HuggingFace hub
    - Insufficient memory to load model
    - Incompatible model architecture
    - Network failure during download
    """


class EncodingError(EncoderError):
    """Raised when encoding operation fails.

    Causes include:
    - Empty or whitespace-only input
    - Input exceeds maximum length (after truncation disabled)
    - Tokenization failure
    - Device memory exhaustion during inference
    """


class ConfigurationError(EncoderError):
    """Raised when encoder configuration is invalid.

    Causes include:
    - Invalid pooling mode specified
    - Dimension mismatch between config and model
    - Invalid device specification
    """
