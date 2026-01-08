"""Configuration schema for Y-Encoder.

Defines immutable configuration using Pydantic v2 with validation
for all encoder parameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class YEncoderConfig(BaseModel):
    """Configuration for Y-Encoder.

    Immutable configuration schema with validation for encoder setup.
    Device resolution follows cuda → mps → cpu fallback chain when
    device is set to 'auto'.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        embedding_dim: Expected output embedding dimension.
        max_length: Maximum input tokens before truncation.
        pooling_mode: Pooling strategy for sequence aggregation.
        normalize: Whether to L2-normalize output embeddings.
        device: Compute device specification.
        cache_size: LRU cache size (0 disables caching).
        trust_remote_code: Allow execution of model-specific code.
    """

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name or local path",
    )
    embedding_dim: int = Field(
        default=384,
        gt=0,
        description="Output embedding dimension",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum input tokens",
    )
    pooling_mode: Literal["mean", "cls", "max"] = Field(
        default="mean",
        description="Pooling strategy: 'mean', 'cls', 'max'",
    )
    normalize: bool = Field(
        default=True,
        description="L2 normalize output embeddings",
    )
    device: Literal["auto", "cuda", "mps", "cpu"] = Field(
        default="auto",
        description="Device for inference",
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="LRU cache size (0 = disabled)",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow execution of model-specific code",
    )

    model_config = {"frozen": True}


# Default configurations for supported models
GEMMA_EMBEDDING_CONFIG = YEncoderConfig(
    model_name="google/gemma-embedding-300m",
    embedding_dim=1024,
    max_length=512,
    pooling_mode="mean",
)

MINILM_CONFIG = YEncoderConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    max_length=256,
    pooling_mode="mean",
)

BGE_LARGE_CONFIG = YEncoderConfig(
    model_name="BAAI/bge-large-en-v1.5",
    embedding_dim=1024,
    max_length=512,
    pooling_mode="cls",
)

QWEN_EMBEDDING_CONFIG = YEncoderConfig(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    embedding_dim=1024,
    max_length=8192,
    pooling_mode="mean",
    trust_remote_code=True,
)
