"""Configuration schema for Semantic State Monitor.

Provides SSMConfig with sensible defaults and validation for
all SSM operating parameters.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SSMConfig(BaseModel):
    """Configuration for Semantic State Monitor.

    Controls the behavior of variance-based boundary detection,
    including window size, thresholds, and smoothing parameters.

    Attributes:
        window_size: Number of recent embeddings to maintain in the
            sliding window. Larger windows provide more stable variance
            but slower response to topic changes. Default: 5.

        variance_threshold: Threshold θ for boundary detection.
            When smoothed variance >= θ, a boundary is detected.
            Default: 0.15.

        smoothing_factor: EMA coefficient α for variance smoothing.
            Higher values make smoothing more responsive, lower values
            provide more stability. Formula: s = α*v + (1-α)*prev.
            Default: 0.3.

        prepare_ratio: Ratio of θ for PREPARE state trigger.
            When variance >= prepare_ratio * θ but < θ, PREPARE is emitted.
            Default: 0.5.

        consolidate_ratio: Ratio of θ for RETRIEVE_CONSOLIDATE trigger.
            When variance >= consolidate_ratio * θ, major topic shift
            is detected. Default: 2.0.

        embedding_dim: Expected embedding dimension from encoder.
            Used for validation. Default: 1024.

    Example:
        >>> config = SSMConfig(
        ...     window_size=10,
        ...     variance_threshold=0.2,
        ...     smoothing_factor=0.4,
        ... )
        >>> ssm = SemanticStateMonitor(encoder, config)
    """

    window_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of turns in sliding window buffer",
    )
    variance_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Threshold θ for boundary detection",
    )
    smoothing_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="EMA smoothing coefficient α",
    )
    prepare_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio of θ for PREPARE state trigger",
    )
    consolidate_ratio: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Ratio of θ for RETRIEVE_CONSOLIDATE trigger",
    )
    embedding_dim: int = Field(
        default=1024,
        ge=1,
        description="Expected embedding dimension from encoder",
    )

    model_config = {"frozen": True}
