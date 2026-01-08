"""Unit tests for predictor configuration classes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prime.predictor import (
    DEFAULT_PREDICTOR_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    FAST_TRAINING_CONFIG,
    LARGE_PREDICTOR_CONFIG,
    SMALL_PREDICTOR_CONFIG,
    PredictorConfig,
    TrainingConfig,
)


class TestPredictorConfig:
    """Tests for PredictorConfig validation."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PredictorConfig()
        assert config.input_dim == 1024
        assert config.hidden_dim == 2048
        assert config.output_dim == 1024
        assert config.num_layers == 4
        assert config.num_heads == 8
        assert config.max_context_length == 10
        assert config.dropout == 0.1
        assert config.checkpoint_path is None
        assert config.use_static_cache is True
        assert config.use_torch_compile is True
        assert config.compile_mode == "reduce-overhead"
        assert config.max_batch_size == 32

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PredictorConfig(
            input_dim=512,
            hidden_dim=1024,
            output_dim=768,
            num_layers=6,
            num_heads=8,
            max_context_length=20,
            dropout=0.2,
            checkpoint_path="/path/to/checkpoint.pt",
            use_static_cache=False,
            use_torch_compile=False,
            compile_mode="max-autotune",
            max_batch_size=16,
        )
        assert config.input_dim == 512
        assert config.hidden_dim == 1024
        assert config.output_dim == 768
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.max_context_length == 20
        assert config.dropout == 0.2
        assert config.checkpoint_path == "/path/to/checkpoint.pt"
        assert config.use_static_cache is False
        assert config.use_torch_compile is False
        assert config.compile_mode == "max-autotune"
        assert config.max_batch_size == 16

    def test_hidden_dim_divisible_by_heads(self) -> None:
        """Test that hidden_dim must be divisible by num_heads."""
        with pytest.raises(ValidationError) as exc_info:
            PredictorConfig(hidden_dim=2047, num_heads=8)
        assert "must be divisible by" in str(exc_info.value)

    def test_hidden_dim_divisible_by_heads_valid(self) -> None:
        """Test valid hidden_dim divisible by num_heads combinations."""
        # Various valid combinations
        config1 = PredictorConfig(hidden_dim=512, num_heads=8)
        assert config1.hidden_dim // config1.num_heads == 64

        config2 = PredictorConfig(hidden_dim=768, num_heads=12)
        assert config2.hidden_dim // config2.num_heads == 64

        config3 = PredictorConfig(hidden_dim=1024, num_heads=16)
        assert config3.hidden_dim // config3.num_heads == 64

    def test_invalid_num_layers(self) -> None:
        """Test invalid num_layers validation."""
        with pytest.raises(ValidationError):
            PredictorConfig(num_layers=0)

        with pytest.raises(ValidationError):
            PredictorConfig(num_layers=13)

    def test_invalid_dropout(self) -> None:
        """Test invalid dropout validation."""
        with pytest.raises(ValidationError):
            PredictorConfig(dropout=-0.1)

        with pytest.raises(ValidationError):
            PredictorConfig(dropout=0.6)

    def test_invalid_max_context_length(self) -> None:
        """Test invalid max_context_length validation."""
        with pytest.raises(ValidationError):
            PredictorConfig(max_context_length=0)

        with pytest.raises(ValidationError):
            PredictorConfig(max_context_length=101)

    def test_invalid_compile_mode(self) -> None:
        """Test invalid compile_mode validation."""
        with pytest.raises(ValidationError):
            PredictorConfig(compile_mode="invalid")  # type: ignore[arg-type]

    def test_frozen_config(self) -> None:
        """Test that config is immutable."""
        config = PredictorConfig()
        with pytest.raises(ValidationError):
            config.hidden_dim = 1024  # type: ignore[misc]

    def test_preset_configs(self) -> None:
        """Test preset configuration instances."""
        # Default config
        assert DEFAULT_PREDICTOR_CONFIG.hidden_dim == 2048
        assert DEFAULT_PREDICTOR_CONFIG.num_layers == 4

        # Small config
        assert SMALL_PREDICTOR_CONFIG.hidden_dim == 1024
        assert SMALL_PREDICTOR_CONFIG.num_layers == 2

        # Large config
        assert LARGE_PREDICTOR_CONFIG.hidden_dim == 4096
        assert LARGE_PREDICTOR_CONFIG.num_layers == 8


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_default_values(self) -> None:
        """Test default training configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.batch_size == 64
        assert config.num_epochs == 10
        assert config.warmup_steps == 1000
        assert config.temperature == 0.07
        assert config.gradient_checkpointing is True
        assert config.mixed_precision is True
        assert config.y_encoder_lr_multiplier == 0.05
        assert config.gradient_clip_val == 1.0
        assert config.accumulate_grad_batches == 1

    def test_custom_values(self) -> None:
        """Test custom training configuration values."""
        config = TrainingConfig(
            learning_rate=3e-4,
            weight_decay=0.1,
            batch_size=128,
            num_epochs=20,
            warmup_steps=500,
            temperature=0.1,
            gradient_checkpointing=False,
            mixed_precision=False,
            y_encoder_lr_multiplier=0.1,
            gradient_clip_val=0.5,
            accumulate_grad_batches=4,
        )
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.1
        assert config.batch_size == 128
        assert config.num_epochs == 20
        assert config.warmup_steps == 500
        assert config.temperature == 0.1
        assert config.gradient_checkpointing is False
        assert config.mixed_precision is False
        assert config.y_encoder_lr_multiplier == 0.1
        assert config.gradient_clip_val == 0.5
        assert config.accumulate_grad_batches == 4

    def test_invalid_learning_rate(self) -> None:
        """Test invalid learning rate validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-1e-4)

    def test_invalid_temperature(self) -> None:
        """Test invalid temperature validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(temperature=0.0)

        with pytest.raises(ValidationError):
            TrainingConfig(temperature=1.5)

    def test_invalid_batch_size(self) -> None:
        """Test invalid batch size validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_invalid_y_encoder_lr_multiplier(self) -> None:
        """Test invalid y_encoder_lr_multiplier validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(y_encoder_lr_multiplier=0.0)

        with pytest.raises(ValidationError):
            TrainingConfig(y_encoder_lr_multiplier=1.5)

    def test_frozen_config(self) -> None:
        """Test that training config is immutable."""
        config = TrainingConfig()
        with pytest.raises(ValidationError):
            config.learning_rate = 1e-3  # type: ignore[misc]

    def test_preset_configs(self) -> None:
        """Test preset training configuration instances."""
        # Default training config
        assert DEFAULT_TRAINING_CONFIG.learning_rate == 1e-4
        assert DEFAULT_TRAINING_CONFIG.num_epochs == 10

        # Fast training config
        assert FAST_TRAINING_CONFIG.learning_rate == 3e-4
        assert FAST_TRAINING_CONFIG.num_epochs == 5
        assert FAST_TRAINING_CONFIG.warmup_steps == 500
