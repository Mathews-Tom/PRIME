"""Y-Encoder module for PRIME.

Provides embedding encoding for target content (responses, documents)
used in memory storage and predictor training.

Public API:
    YEncoder: Main encoder implementation
    YEncoderConfig: Configuration schema for encoder setup
    Encoder: Protocol defining encoder interface
    EncoderError: Base exception for encoder operations
    ModelLoadError: Raised when model fails to load
    EncodingError: Raised when encoding operation fails
    ConfigurationError: Raised when configuration is invalid

Example:
    >>> from prime.encoder import YEncoder, YEncoderConfig
    >>> config = YEncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    >>> encoder = YEncoder(config)
    >>> embedding = encoder.encode("Hello, world!")
"""

from __future__ import annotations

from prime.encoder.config import (
    BGE_LARGE_CONFIG,
    GEMMA_EMBEDDING_CONFIG,
    MINILM_CONFIG,
    QWEN_EMBEDDING_CONFIG,
    YEncoderConfig,
)
from prime.encoder.exceptions import (
    ConfigurationError,
    EncoderError,
    EncodingError,
    ModelLoadError,
)
from prime.encoder.pooling import PoolingMode, pool_embeddings
from prime.encoder.protocols import Encoder
from prime.encoder.y_encoder import CacheInfo, YEncoder

__all__ = [
    "BGE_LARGE_CONFIG",
    "GEMMA_EMBEDDING_CONFIG",
    "MINILM_CONFIG",
    "QWEN_EMBEDDING_CONFIG",
    "CacheInfo",
    "ConfigurationError",
    "Encoder",
    "EncoderError",
    "EncodingError",
    "ModelLoadError",
    "PoolingMode",
    "YEncoder",
    "YEncoderConfig",
    "pool_embeddings",
]
