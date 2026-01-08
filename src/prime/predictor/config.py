"""Configuration schema for Embedding Predictor.

Defines immutable configuration using Pydantic v2 with validation
for all predictor parameters including model architecture, optimization,
and training settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PredictorConfig(BaseModel):
    """Configuration for Embedding Predictor.

    Immutable configuration schema with validation for predictor setup.
    Includes architecture parameters, optimization settings, and
    Static KV Cache configuration for production inference.

    Attributes:
        input_dim: Input embedding dimension (match X-Encoder).
        hidden_dim: Transformer hidden dimension.
        output_dim: Output embedding dimension (match Y-Encoder).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_context_length: Maximum context turns supported.
        dropout: Dropout probability.
        checkpoint_path: Path to model checkpoint.
        use_static_cache: Enable Static KV Cache for 4x inference speedup.
        use_torch_compile: Enable torch.compile() for CUDA graph capture.
        compile_mode: torch.compile optimization mode.
        max_batch_size: Maximum batch size for static shape compilation.
    """

    input_dim: int = Field(
        default=1024,
        gt=0,
        description="Input embedding dimension (match X-Encoder)",
    )
    hidden_dim: int = Field(
        default=2048,
        gt=0,
        description="Transformer hidden dimension",
    )
    output_dim: int = Field(
        default=1024,
        gt=0,
        description="Output embedding dimension (match Y-Encoder)",
    )
    num_layers: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of transformer layers",
    )
    num_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads",
    )
    max_context_length: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum context turns",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout probability",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to model checkpoint",
    )

    # Static KV Cache Configuration
    use_static_cache: bool = Field(
        default=True,
        description="Enable Static KV Cache for 4x faster inference",
    )
    use_torch_compile: bool = Field(
        default=True,
        description="Enable torch.compile() for CUDA graph capture",
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
        description="torch.compile optimization mode",
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum batch size for static shape compilation",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_hidden_dim_divisible_by_heads(self) -> PredictorConfig:
        """Ensure hidden_dim is divisible by num_heads for attention."""
        if self.hidden_dim % self.num_heads != 0:
            msg = (
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
            raise ValueError(msg)
        return self


class TrainingConfig(BaseModel):
    """Training configuration for Embedding Predictor.

    Immutable configuration for training loop parameters including
    learning rate schedules, regularization, and optimization settings.

    Attributes:
        learning_rate: Base learning rate for AdamW optimizer.
        weight_decay: L2 regularization weight.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        warmup_steps: Linear warmup steps before decay.
        temperature: InfoNCE temperature parameter (tau).
        gradient_checkpointing: Enable gradient checkpointing for memory.
        mixed_precision: Enable automatic mixed precision training.
        y_encoder_lr_multiplier: LR multiplier for Y-Encoder (slower update).
        gradient_clip_val: Maximum gradient norm for clipping.
        accumulate_grad_batches: Gradient accumulation steps.
    """

    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Base learning rate",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay (L2 regularization)",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Training batch size",
    )
    num_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of training epochs",
    )
    warmup_steps: int = Field(
        default=1000,
        ge=0,
        description="Linear warmup steps",
    )
    temperature: float = Field(
        default=0.07,
        gt=0.0,
        le=1.0,
        description="InfoNCE temperature (tau)",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing for memory efficiency",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable automatic mixed precision (AMP)",
    )
    y_encoder_lr_multiplier: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Learning rate multiplier for Y-Encoder",
    )
    gradient_clip_val: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for clipping",
    )
    accumulate_grad_batches: int = Field(
        default=1,
        ge=1,
        description="Number of batches for gradient accumulation",
    )

    model_config = {"frozen": True}


# Default configurations for common use cases
DEFAULT_PREDICTOR_CONFIG = PredictorConfig()

SMALL_PREDICTOR_CONFIG = PredictorConfig(
    hidden_dim=1024,
    num_layers=2,
    num_heads=4,
)

LARGE_PREDICTOR_CONFIG = PredictorConfig(
    hidden_dim=4096,
    num_layers=8,
    num_heads=16,
)

DEFAULT_TRAINING_CONFIG = TrainingConfig()

FAST_TRAINING_CONFIG = TrainingConfig(
    learning_rate=3e-4,
    num_epochs=5,
    warmup_steps=500,
)
