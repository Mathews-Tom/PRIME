"""Exception hierarchy for the predictor module.

Defines specific exception types for predictor operations following
the fail-fast principle with explicit error handling.
"""

from __future__ import annotations


class PredictorError(Exception):
    """Base exception for all predictor-related errors.

    All predictor exceptions inherit from this class, enabling
    catch-all handling when needed while preserving specificity.
    """


class CheckpointError(PredictorError):
    """Raised when checkpoint operations fail.

    Causes include:
    - Checkpoint file not found at specified path
    - Corrupted or incompatible checkpoint format
    - State dict key mismatch during loading
    - Insufficient disk space during saving
    """


class PredictorShapeError(PredictorError):
    """Raised when tensor shapes are invalid.

    Causes include:
    - Context embeddings dimension mismatch
    - Query embedding dimension mismatch
    - Batch size exceeds configured maximum
    - Context length exceeds configured maximum
    """


class PredictorConfigError(PredictorError):
    """Raised when predictor configuration is invalid.

    Causes include:
    - Invalid hidden dimension (not divisible by num_heads)
    - Invalid num_layers or num_heads values
    - Compile mode not supported by PyTorch version
    - Device not available
    """


class OptimizationError(PredictorError):
    """Raised when optimization features fail.

    Causes include:
    - Static KV Cache initialization failure
    - torch.compile() compilation failure
    - CUDA graph capture failure
    - Memory allocation failure for static tensors
    """


class TrainingError(PredictorError):
    """Raised when training operations fail.

    Causes include:
    - Loss computation failure (NaN/Inf)
    - Gradient explosion
    - Learning rate scheduler error
    - Data loader exhaustion
    """
