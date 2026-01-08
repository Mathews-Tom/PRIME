"""Unit tests for pooling functions."""

from __future__ import annotations

import pytest
import torch

from prime.encoder.pooling import (
    PoolingMode,
    pool_embeddings,
)


class TestPoolEmbeddings:
    """Tests for pool_embeddings dispatcher function."""

    @pytest.fixture
    def sample_hidden_states(self) -> torch.Tensor:
        """Sample hidden states: batch=2, seq=4, hidden=8."""
        torch.manual_seed(42)
        return torch.randn(2, 4, 8)

    @pytest.fixture
    def full_attention_mask(self) -> torch.Tensor:
        """Attention mask with all tokens valid."""
        return torch.ones(2, 4)

    @pytest.fixture
    def partial_attention_mask(self) -> torch.Tensor:
        """Attention mask with some padding."""
        return torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    def test_mean_pooling_output_shape(
        self,
        sample_hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> None:
        """Mean pooling produces correct output shape."""
        result = pool_embeddings(sample_hidden_states, full_attention_mask, "mean")
        assert result.shape == (2, 8)

    def test_cls_pooling_output_shape(
        self,
        sample_hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> None:
        """CLS pooling produces correct output shape."""
        result = pool_embeddings(sample_hidden_states, full_attention_mask, "cls")
        assert result.shape == (2, 8)

    def test_max_pooling_output_shape(
        self,
        sample_hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> None:
        """Max pooling produces correct output shape."""
        result = pool_embeddings(sample_hidden_states, full_attention_mask, "max")
        assert result.shape == (2, 8)

    def test_invalid_pooling_mode_raises(
        self,
        sample_hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> None:
        """Invalid pooling mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pooling mode"):
            pool_embeddings(
                sample_hidden_states,
                full_attention_mask,
                "invalid",  # type: ignore[arg-type]
            )

    def test_cls_pooling_returns_first_token(
        self,
        sample_hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> None:
        """CLS pooling returns exactly the first token."""
        result = pool_embeddings(sample_hidden_states, full_attention_mask, "cls")
        expected = sample_hidden_states[:, 0, :]
        assert torch.allclose(result, expected)

    def test_mean_pooling_respects_attention_mask(
        self,
        sample_hidden_states: torch.Tensor,
        partial_attention_mask: torch.Tensor,
    ) -> None:
        """Mean pooling correctly excludes masked tokens."""
        result = pool_embeddings(sample_hidden_states, partial_attention_mask, "mean")

        # Manual calculation for first sample (3 valid tokens)
        valid_tokens = sample_hidden_states[0, :3, :]
        expected_first = valid_tokens.mean(dim=0)
        assert torch.allclose(result[0], expected_first, atol=1e-5)

        # Second sample (2 valid tokens)
        valid_tokens = sample_hidden_states[1, :2, :]
        expected_second = valid_tokens.mean(dim=0)
        assert torch.allclose(result[1], expected_second, atol=1e-5)

    def test_max_pooling_respects_attention_mask(
        self,
        sample_hidden_states: torch.Tensor,
        partial_attention_mask: torch.Tensor,
    ) -> None:
        """Max pooling correctly excludes masked tokens."""
        result = pool_embeddings(sample_hidden_states, partial_attention_mask, "max")

        # Manual calculation for first sample (3 valid tokens)
        valid_tokens = sample_hidden_states[0, :3, :]
        expected_first = valid_tokens.max(dim=0)[0]
        assert torch.allclose(result[0], expected_first)

    @pytest.mark.parametrize("mode", ["mean", "cls", "max"])
    def test_pooling_modes_deterministic(
        self,
        sample_hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
        mode: PoolingMode,
    ) -> None:
        """Pooling operations are deterministic."""
        result1 = pool_embeddings(sample_hidden_states, full_attention_mask, mode)
        result2 = pool_embeddings(sample_hidden_states, full_attention_mask, mode)
        assert torch.equal(result1, result2)

    def test_mean_pooling_single_token(self) -> None:
        """Mean pooling with single token returns that token."""
        hidden = torch.tensor([[[1.0, 2.0, 3.0]]])
        mask = torch.tensor([[1]])
        result = pool_embeddings(hidden, mask, "mean")
        assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_max_pooling_negative_values(self) -> None:
        """Max pooling correctly handles negative values."""
        hidden = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]]])
        mask = torch.tensor([[1, 1]])
        result = pool_embeddings(hidden, mask, "max")
        assert torch.allclose(result, torch.tensor([[-1.0, -2.0]]))
