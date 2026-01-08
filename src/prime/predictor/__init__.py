"""Embedding Predictor module for PRIME.

Provides JEPA-style embedding prediction for targeted retrieval.
The predictor learns to predict the embedding of ideal context
BEFORE retrieval, enabling more targeted search.

Public API:
    EmbeddingPredictor: Core predictor nn.Module
    PredictorTransformerBlock: Transformer block with pre-norm
    PredictorConfig: Configuration schema for predictor setup
    TrainingConfig: Configuration for training parameters
    create_predictor: Factory function to create predictor
    PredictorError: Base exception for predictor operations
    CheckpointError: Raised when checkpoint operations fail
    PredictorShapeError: Raised when tensor shapes are invalid
    PredictorConfigError: Raised when configuration is invalid
    OptimizationError: Raised when optimization features fail
    TrainingError: Raised when training operations fail
    PredictorInput: Input data class for prediction
    PredictorOutput: Output data class from prediction
    TrainingBatch: Batch of training samples
    TrainingMetrics: Metrics from training step

Example:
    >>> from prime.predictor import EmbeddingPredictor, PredictorConfig
    >>> config = PredictorConfig(hidden_dim=2048, num_layers=4)
    >>> predictor = EmbeddingPredictor(config)
"""

from __future__ import annotations

from prime.predictor.config import (
    DEFAULT_PREDICTOR_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    FAST_TRAINING_CONFIG,
    LARGE_PREDICTOR_CONFIG,
    SMALL_PREDICTOR_CONFIG,
    PredictorConfig,
    TrainingConfig,
)
from prime.predictor.exceptions import (
    CheckpointError,
    OptimizationError,
    PredictorConfigError,
    PredictorError,
    PredictorShapeError,
    TrainingError,
)
from prime.predictor.optimized import (
    OptimizedPredictor,
    create_optimized_predictor,
)
from prime.predictor.predictor import (
    EmbeddingPredictor,
    PredictorTransformerBlock,
    create_predictor,
)
from prime.predictor.types import (
    CheckpointMetadata,
    PredictorInput,
    PredictorOutput,
    TrainingBatch,
    TrainingMetrics,
)

__all__ = [
    # Configuration
    "DEFAULT_PREDICTOR_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    # Core
    "EmbeddingPredictor",
    "FAST_TRAINING_CONFIG",
    "LARGE_PREDICTOR_CONFIG",
    # Optimization
    "OptimizedPredictor",
    "SMALL_PREDICTOR_CONFIG",
    # Exceptions
    "CheckpointError",
    # Types
    "CheckpointMetadata",
    "OptimizationError",
    "PredictorConfig",
    "PredictorConfigError",
    "PredictorError",
    "PredictorInput",
    "PredictorOutput",
    "PredictorShapeError",
    "PredictorTransformerBlock",
    "TrainingBatch",
    "TrainingConfig",
    "TrainingError",
    "TrainingMetrics",
    "create_optimized_predictor",
    "create_predictor",
]
