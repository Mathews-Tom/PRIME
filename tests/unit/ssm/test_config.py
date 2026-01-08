"""Tests for SSMConfig validation and immutability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prime.ssm import SSMConfig


class TestSSMConfigDefaults:
    """Test SSMConfig default values."""

    def test_default_window_size(self) -> None:
        """Default window_size is 5."""
        config = SSMConfig()
        assert config.window_size == 5

    def test_default_variance_threshold(self) -> None:
        """Default variance_threshold is 0.15."""
        config = SSMConfig()
        assert config.variance_threshold == 0.15

    def test_default_smoothing_factor(self) -> None:
        """Default smoothing_factor is 0.3."""
        config = SSMConfig()
        assert config.smoothing_factor == 0.3

    def test_default_prepare_ratio(self) -> None:
        """Default prepare_ratio is 0.5."""
        config = SSMConfig()
        assert config.prepare_ratio == 0.5

    def test_default_consolidate_ratio(self) -> None:
        """Default consolidate_ratio is 2.0."""
        config = SSMConfig()
        assert config.consolidate_ratio == 2.0

    def test_default_embedding_dim(self) -> None:
        """Default embedding_dim is 1024."""
        config = SSMConfig()
        assert config.embedding_dim == 1024


class TestSSMConfigValidation:
    """Test SSMConfig field validation."""

    def test_window_size_minimum(self) -> None:
        """window_size must be >= 1."""
        with pytest.raises(ValidationError):
            SSMConfig(window_size=0)

    def test_window_size_maximum(self) -> None:
        """window_size must be <= 50."""
        with pytest.raises(ValidationError):
            SSMConfig(window_size=51)

    def test_window_size_valid_range(self) -> None:
        """window_size accepts valid range values."""
        assert SSMConfig(window_size=1).window_size == 1
        assert SSMConfig(window_size=50).window_size == 50

    def test_variance_threshold_minimum(self) -> None:
        """variance_threshold must be >= 0.0."""
        with pytest.raises(ValidationError):
            SSMConfig(variance_threshold=-0.1)

    def test_variance_threshold_maximum(self) -> None:
        """variance_threshold must be <= 1.0."""
        with pytest.raises(ValidationError):
            SSMConfig(variance_threshold=1.1)

    def test_variance_threshold_valid_range(self) -> None:
        """variance_threshold accepts valid range values."""
        assert SSMConfig(variance_threshold=0.0).variance_threshold == 0.0
        assert SSMConfig(variance_threshold=1.0).variance_threshold == 1.0

    def test_smoothing_factor_minimum(self) -> None:
        """smoothing_factor must be >= 0.0."""
        with pytest.raises(ValidationError):
            SSMConfig(smoothing_factor=-0.1)

    def test_smoothing_factor_maximum(self) -> None:
        """smoothing_factor must be <= 1.0."""
        with pytest.raises(ValidationError):
            SSMConfig(smoothing_factor=1.1)

    def test_prepare_ratio_minimum(self) -> None:
        """prepare_ratio must be >= 0.0."""
        with pytest.raises(ValidationError):
            SSMConfig(prepare_ratio=-0.1)

    def test_prepare_ratio_maximum(self) -> None:
        """prepare_ratio must be <= 1.0."""
        with pytest.raises(ValidationError):
            SSMConfig(prepare_ratio=1.1)

    def test_consolidate_ratio_minimum(self) -> None:
        """consolidate_ratio must be >= 1.0."""
        with pytest.raises(ValidationError):
            SSMConfig(consolidate_ratio=0.9)

    def test_consolidate_ratio_maximum(self) -> None:
        """consolidate_ratio must be <= 10.0."""
        with pytest.raises(ValidationError):
            SSMConfig(consolidate_ratio=10.1)

    def test_embedding_dim_minimum(self) -> None:
        """embedding_dim must be >= 1."""
        with pytest.raises(ValidationError):
            SSMConfig(embedding_dim=0)


class TestSSMConfigImmutability:
    """Test SSMConfig is immutable (frozen)."""

    def test_window_size_immutable(self) -> None:
        """Cannot modify window_size after creation."""
        config = SSMConfig()
        with pytest.raises(ValidationError):
            config.window_size = 10  # type: ignore[misc]

    def test_variance_threshold_immutable(self) -> None:
        """Cannot modify variance_threshold after creation."""
        config = SSMConfig()
        with pytest.raises(ValidationError):
            config.variance_threshold = 0.5  # type: ignore[misc]

    def test_smoothing_factor_immutable(self) -> None:
        """Cannot modify smoothing_factor after creation."""
        config = SSMConfig()
        with pytest.raises(ValidationError):
            config.smoothing_factor = 0.5  # type: ignore[misc]


class TestSSMConfigEquality:
    """Test SSMConfig equality comparison."""

    def test_equal_configs(self) -> None:
        """Configs with same values are equal."""
        config1 = SSMConfig(window_size=5)
        config2 = SSMConfig(window_size=5)
        assert config1 == config2

    def test_unequal_configs(self) -> None:
        """Configs with different values are not equal."""
        config1 = SSMConfig(window_size=5)
        config2 = SSMConfig(window_size=10)
        assert config1 != config2
