"""Type definitions for the predictor module.

Defines data classes, named tuples, and type aliases used throughout
the predictor implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PredictorInput:
    """Input for embedding prediction.

    Attributes:
        context_embeddings: Context turn embeddings (N x D).
        query_embedding: Current query embedding (D,).
    """

    context_embeddings: NDArray[np.float32]
    query_embedding: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate input shapes."""
        if self.context_embeddings.ndim != 2:
            msg = (
                f"context_embeddings must be 2D (N x D), "
                f"got shape {self.context_embeddings.shape}"
            )
            raise ValueError(msg)
        if self.query_embedding.ndim != 1:
            msg = (
                f"query_embedding must be 1D (D,), "
                f"got shape {self.query_embedding.shape}"
            )
            raise ValueError(msg)
        if self.context_embeddings.shape[1] != self.query_embedding.shape[0]:
            msg = (
                f"Dimension mismatch: context has {self.context_embeddings.shape[1]}, "
                f"query has {self.query_embedding.shape[0]}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PredictorOutput:
    """Output from embedding prediction.

    Attributes:
        predicted_embedding: Predicted target embedding (D,).
        confidence: Prediction confidence score (optional metric).
    """

    predicted_embedding: NDArray[np.float32]
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Batch of training samples for predictor.

    Attributes:
        context_embeddings: Batch of context embeddings (B x N x D).
        query_embeddings: Batch of query embeddings (B x D).
        target_embeddings: Batch of target embeddings from Y-Encoder (B x D).
    """

    context_embeddings: NDArray[np.float32]
    query_embeddings: NDArray[np.float32]
    target_embeddings: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Metrics from a training step.

    Attributes:
        loss: InfoNCE loss value.
        accuracy: Top-1 accuracy (predicted vs target).
        learning_rate: Current learning rate.
        gradient_norm: Gradient L2 norm before clipping.
    """

    loss: float
    accuracy: float
    learning_rate: float
    gradient_norm: float | None = None


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Metadata stored with model checkpoints.

    Attributes:
        epoch: Training epoch when checkpoint was saved.
        step: Global step count.
        loss: Validation loss at checkpoint.
        config_hash: Hash of PredictorConfig for compatibility check.
    """

    epoch: int
    step: int
    loss: float
    config_hash: str


# Type alias for embedding arrays
EmbeddingArray = NDArray[np.float32]
