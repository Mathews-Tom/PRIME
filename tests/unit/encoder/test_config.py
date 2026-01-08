"""Unit tests for YEncoderConfig validation.

Tests configuration schema, defaults, validation rules, and immutability.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prime.encoder import (
    BGE_LARGE_CONFIG,
    GEMMA_EMBEDDING_CONFIG,
    MINILM_CONFIG,
    QWEN_EMBEDDING_CONFIG,
    YEncoderConfig,
)


class TestYEncoderConfigDefaults:
    """Tests for default configuration values."""

    def test_default_model_name(self) -> None:
        """Default model is MiniLM for fast testing."""
        config = YEncoderConfig()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_default_embedding_dim(self) -> None:
        """Default dimension matches MiniLM output."""
        config = YEncoderConfig()
        assert config.embedding_dim == 384

    def test_default_max_length(self) -> None:
        """Default max length is 512 tokens."""
        config = YEncoderConfig()
        assert config.max_length == 512

    def test_default_pooling_mode(self) -> None:
        """Default pooling is mean pooling."""
        config = YEncoderConfig()
        assert config.pooling_mode == "mean"

    def test_default_normalize(self) -> None:
        """L2 normalization is enabled by default."""
        config = YEncoderConfig()
        assert config.normalize is True

    def test_default_device(self) -> None:
        """Device defaults to auto-detection."""
        config = YEncoderConfig()
        assert config.device == "auto"

    def test_default_cache_disabled(self) -> None:
        """Caching is disabled by default."""
        config = YEncoderConfig()
        assert config.cache_size == 0

    def test_default_trust_remote_code(self) -> None:
        """Remote code execution is disabled by default."""
        config = YEncoderConfig()
        assert config.trust_remote_code is False


class TestYEncoderConfigValidation:
    """Tests for configuration validation rules."""

    def test_embedding_dim_must_be_positive(self) -> None:
        """Embedding dimension must be greater than zero."""
        with pytest.raises(ValidationError):
            YEncoderConfig(embedding_dim=0)

        with pytest.raises(ValidationError):
            YEncoderConfig(embedding_dim=-1)

    def test_max_length_minimum(self) -> None:
        """Max length must be at least 1."""
        with pytest.raises(ValidationError):
            YEncoderConfig(max_length=0)

    def test_max_length_maximum(self) -> None:
        """Max length cannot exceed 8192."""
        with pytest.raises(ValidationError):
            YEncoderConfig(max_length=8193)

    def test_max_length_valid_boundary(self) -> None:
        """Max length at boundaries is valid."""
        config_min = YEncoderConfig(max_length=1)
        assert config_min.max_length == 1

        config_max = YEncoderConfig(max_length=8192)
        assert config_max.max_length == 8192

    def test_pooling_mode_valid_values(self) -> None:
        """Valid pooling modes are accepted."""
        for mode in ["mean", "cls", "max"]:
            config = YEncoderConfig(pooling_mode=mode)  # type: ignore[arg-type]
            assert config.pooling_mode == mode

    def test_pooling_mode_invalid_value(self) -> None:
        """Invalid pooling mode raises error."""
        with pytest.raises(ValidationError):
            YEncoderConfig(pooling_mode="average")  # type: ignore[arg-type]

    def test_device_valid_values(self) -> None:
        """Valid device specifications are accepted."""
        for device in ["auto", "cuda", "mps", "cpu"]:
            config = YEncoderConfig(device=device)  # type: ignore[arg-type]
            assert config.device == device

    def test_device_invalid_value(self) -> None:
        """Invalid device raises error."""
        with pytest.raises(ValidationError):
            YEncoderConfig(device="gpu")  # type: ignore[arg-type]

    def test_cache_size_non_negative(self) -> None:
        """Cache size must be non-negative."""
        with pytest.raises(ValidationError):
            YEncoderConfig(cache_size=-1)

    def test_cache_size_zero_valid(self) -> None:
        """Cache size of zero (disabled) is valid."""
        config = YEncoderConfig(cache_size=0)
        assert config.cache_size == 0


class TestYEncoderConfigImmutability:
    """Tests for configuration immutability."""

    def test_config_is_frozen(self) -> None:
        """Configuration cannot be modified after creation."""
        config = YEncoderConfig()

        with pytest.raises(ValidationError):
            config.model_name = "different-model"  # type: ignore[misc]

    def test_config_hashable(self) -> None:
        """Frozen config is hashable for use in sets/dicts."""
        config1 = YEncoderConfig()
        config2 = YEncoderConfig()

        # Same configs should have same hash
        assert hash(config1) == hash(config2)

        # Can be used in sets
        config_set = {config1, config2}
        assert len(config_set) == 1


class TestPresetConfigurations:
    """Tests for preset model configurations."""

    def test_minilm_config(self) -> None:
        """MiniLM preset has correct values."""
        assert MINILM_CONFIG.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert MINILM_CONFIG.embedding_dim == 384
        assert MINILM_CONFIG.max_length == 256
        assert MINILM_CONFIG.pooling_mode == "mean"

    def test_gemma_config(self) -> None:
        """Gemma Embedding preset has correct values."""
        assert GEMMA_EMBEDDING_CONFIG.model_name == "google/gemma-embedding-300m"
        assert GEMMA_EMBEDDING_CONFIG.embedding_dim == 1024
        assert GEMMA_EMBEDDING_CONFIG.max_length == 512
        assert GEMMA_EMBEDDING_CONFIG.pooling_mode == "mean"

    def test_bge_large_config(self) -> None:
        """BGE Large preset has correct values."""
        assert BGE_LARGE_CONFIG.model_name == "BAAI/bge-large-en-v1.5"
        assert BGE_LARGE_CONFIG.embedding_dim == 1024
        assert BGE_LARGE_CONFIG.max_length == 512
        assert BGE_LARGE_CONFIG.pooling_mode == "cls"

    def test_qwen_config(self) -> None:
        """Qwen Embedding preset has correct values."""
        assert QWEN_EMBEDDING_CONFIG.model_name == "Qwen/Qwen3-Embedding-0.6B"
        assert QWEN_EMBEDDING_CONFIG.embedding_dim == 1024
        assert QWEN_EMBEDDING_CONFIG.max_length == 8192
        assert QWEN_EMBEDDING_CONFIG.pooling_mode == "mean"
        assert QWEN_EMBEDDING_CONFIG.trust_remote_code is True


class TestYEncoderConfigEquality:
    """Tests for configuration equality comparison."""

    def test_equal_configs(self) -> None:
        """Configs with same values are equal."""
        config1 = YEncoderConfig(model_name="test", embedding_dim=512)
        config2 = YEncoderConfig(model_name="test", embedding_dim=512)
        assert config1 == config2

    def test_different_configs(self) -> None:
        """Configs with different values are not equal."""
        config1 = YEncoderConfig(model_name="model-a")
        config2 = YEncoderConfig(model_name="model-b")
        assert config1 != config2
