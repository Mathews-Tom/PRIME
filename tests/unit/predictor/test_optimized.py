"""Tests for OptimizedPredictor.

Tests torch.compile wrapper and static batch padding.
"""

from __future__ import annotations

import pytest
import torch

from prime.predictor import (
    EmbeddingPredictor,
    OptimizedPredictor,
    PredictorConfig,
    PredictorError,
    create_optimized_predictor,
)


class TestOptimizedPredictorInit:
    """Tests for OptimizedPredictor initialization."""

    def test_init_with_defaults(self) -> None:
        """Should initialize with default config."""
        opt = OptimizedPredictor()

        assert opt.config is not None
        assert opt.predictor is not None
        assert not opt.is_warmed_up

    def test_init_with_custom_config(self) -> None:
        """Should use provided config."""
        config = PredictorConfig(hidden_dim=512, num_layers=2)
        opt = OptimizedPredictor(config)

        assert opt.config.hidden_dim == 512
        assert opt.config.num_layers == 2

    def test_init_with_existing_predictor(self) -> None:
        """Should wrap existing predictor."""
        config = PredictorConfig(hidden_dim=256)
        predictor = EmbeddingPredictor(config)
        opt = OptimizedPredictor(config, predictor=predictor)

        assert opt.predictor is predictor

    def test_init_moves_to_device(self) -> None:
        """Should move predictor to specified device."""
        opt = OptimizedPredictor(device="cpu")

        # Check predictor is on CPU
        param = next(opt.predictor.parameters())
        assert param.device.type == "cpu"

    def test_init_sets_eval_mode(self) -> None:
        """Should set predictor to eval mode."""
        opt = OptimizedPredictor()

        assert not opt.predictor.training


class TestOptimizedPredictorProperties:
    """Tests for OptimizedPredictor properties."""

    def test_predictor_property(self) -> None:
        """predictor should return underlying EmbeddingPredictor."""
        opt = OptimizedPredictor()

        assert isinstance(opt.predictor, EmbeddingPredictor)

    def test_is_warmed_up_property(self) -> None:
        """is_warmed_up should track warmup state."""
        opt = OptimizedPredictor()

        assert not opt.is_warmed_up

    def test_device_property(self) -> None:
        """device should return correct device."""
        opt = OptimizedPredictor(device="cpu")

        assert opt.device.type == "cpu"


class TestOptimizedPredictorWarmup:
    """Tests for warmup method."""

    def test_warmup_sets_flag(self) -> None:
        """warmup should set is_warmed_up flag."""
        opt = OptimizedPredictor()
        opt.warmup()

        assert opt.is_warmed_up

    def test_warmup_idempotent(self) -> None:
        """Multiple warmup calls should be safe."""
        opt = OptimizedPredictor()
        opt.warmup()
        opt.warmup()  # Should not error

        assert opt.is_warmed_up


class TestOptimizedPredictorPadding:
    """Tests for static batch padding methods."""

    @pytest.fixture
    def opt(self) -> OptimizedPredictor:
        """Create OptimizedPredictor for testing."""
        config = PredictorConfig(max_batch_size=8, max_context_length=16)
        return OptimizedPredictor(config)

    def test_pad_to_static_batch_smaller(self, opt: OptimizedPredictor) -> None:
        """Should pad smaller batch to target size."""
        tensor = torch.randn(4, 256)
        padded = opt._pad_to_static_batch(tensor, 8)

        assert padded.shape == (8, 256)
        # Original values preserved
        assert torch.allclose(padded[:4], tensor)
        # Padding is zeros
        assert torch.allclose(padded[4:], torch.zeros(4, 256))

    def test_pad_to_static_batch_equal(self, opt: OptimizedPredictor) -> None:
        """Should return unchanged when batch equals target."""
        tensor = torch.randn(8, 256)
        padded = opt._pad_to_static_batch(tensor, 8)

        assert padded.shape == (8, 256)
        assert torch.allclose(padded, tensor)

    def test_pad_to_static_batch_larger(self, opt: OptimizedPredictor) -> None:
        """Should truncate when batch exceeds target."""
        tensor = torch.randn(10, 256)
        padded = opt._pad_to_static_batch(tensor, 8)

        assert padded.shape == (8, 256)
        assert torch.allclose(padded, tensor[:8])

    def test_pad_context_smaller(self, opt: OptimizedPredictor) -> None:
        """Should pad shorter context sequence."""
        context = torch.randn(2, 8, 256)
        padded = opt._pad_context(context, 16)

        assert padded.shape == (2, 16, 256)
        # Original values preserved
        assert torch.allclose(padded[:, :8, :], context)

    def test_pad_context_equal(self, opt: OptimizedPredictor) -> None:
        """Should return unchanged when context equals target."""
        context = torch.randn(2, 16, 256)
        padded = opt._pad_context(context, 16)

        assert padded.shape == (2, 16, 256)
        assert torch.allclose(padded, context)

    def test_pad_context_larger(self, opt: OptimizedPredictor) -> None:
        """Should truncate when context exceeds target."""
        context = torch.randn(2, 20, 256)
        padded = opt._pad_context(context, 16)

        assert padded.shape == (2, 16, 256)
        assert torch.allclose(padded, context[:, :16, :])


class TestOptimizedPredictorForward:
    """Tests for forward pass."""

    @pytest.fixture
    def opt(self) -> OptimizedPredictor:
        """Create OptimizedPredictor for testing."""
        config = PredictorConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            max_batch_size=8,
            max_context_length=16,
        )
        return OptimizedPredictor(config, device="cpu")

    def test_forward_output_shape(self, opt: OptimizedPredictor) -> None:
        """Output should match batch size and output dim."""
        context = torch.randn(4, 8, 256)
        query = torch.randn(4, 256)

        output = opt(context, query)

        assert output.shape == (4, 256)

    def test_forward_single_sample(self, opt: OptimizedPredictor) -> None:
        """Should work with single sample."""
        context = torch.randn(1, 4, 256)
        query = torch.randn(1, 256)

        output = opt(context, query)

        assert output.shape == (1, 256)

    def test_forward_preserves_batch_size(self, opt: OptimizedPredictor) -> None:
        """Output batch size should match input."""
        context = torch.randn(3, 8, 256)
        query = torch.randn(3, 256)

        output = opt(context, query)

        # Original batch preserved even with padding
        assert output.shape[0] == 3

    def test_forward_invalid_context_ndim(self, opt: OptimizedPredictor) -> None:
        """Should raise for invalid context dimensions."""
        context = torch.randn(4, 256)  # Missing sequence dim
        query = torch.randn(4, 256)

        with pytest.raises(PredictorError, match="context_embeddings must be 3D"):
            opt(context, query)

    def test_forward_invalid_query_ndim(self, opt: OptimizedPredictor) -> None:
        """Should raise for invalid query dimensions."""
        context = torch.randn(4, 8, 256)
        query = torch.randn(4, 8, 256)  # Too many dims

        with pytest.raises(PredictorError, match="query_embedding must be 2D"):
            opt(context, query)

    def test_forward_method_alias(self, opt: OptimizedPredictor) -> None:
        """forward() should be alias for __call__."""
        context = torch.randn(2, 4, 256)
        query = torch.randn(2, 256)

        output_call = opt(context, query)
        output_forward = opt.forward(context, query)

        # Both should produce same shape
        assert output_call.shape == output_forward.shape


class TestOptimizedPredictorDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Same input should produce same output in eval mode."""
        config = PredictorConfig(input_dim=128, hidden_dim=256, output_dim=128)
        opt = OptimizedPredictor(config, device="cpu")

        context = torch.randn(2, 4, 128)
        query = torch.randn(2, 128)

        output1 = opt(context, query)
        output2 = opt(context, query)

        assert torch.allclose(output1, output2)


class TestCreateOptimizedPredictor:
    """Tests for factory function."""

    def test_creates_optimized_predictor(self) -> None:
        """Should return OptimizedPredictor instance."""
        opt = create_optimized_predictor()

        assert isinstance(opt, OptimizedPredictor)

    def test_with_config(self) -> None:
        """Should use provided config."""
        config = PredictorConfig(hidden_dim=256)
        opt = create_optimized_predictor(config)

        assert opt.config.hidden_dim == 256

    def test_with_device(self) -> None:
        """Should set device."""
        opt = create_optimized_predictor(device="cpu")

        assert opt.device.type == "cpu"

    def test_with_predictor(self) -> None:
        """Should wrap existing predictor."""
        config = PredictorConfig()
        predictor = EmbeddingPredictor(config)
        opt = create_optimized_predictor(predictor=predictor)

        assert opt.predictor is predictor
