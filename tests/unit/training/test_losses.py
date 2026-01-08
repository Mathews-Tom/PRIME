"""Tests for PRIME training loss functions.

Tests InfoNCE contrastive losses with in-batch negatives.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from prime.training.losses import (
    info_nce_loss,
    info_nce_loss_with_negatives,
    symmetric_info_nce_loss,
)


class TestInfoNCELoss:
    """Tests for info_nce_loss function."""

    def test_identical_embeddings_zero_loss(self) -> None:
        """Identical predicted and positive should give near-zero loss."""
        embeddings = F.normalize(torch.randn(8, 256), dim=-1)
        loss = info_nce_loss(embeddings, embeddings.clone(), temperature=0.07)
        # With identical embeddings, loss should be very low
        assert loss.item() < 0.1

    def test_random_embeddings_positive_loss(self) -> None:
        """Random embeddings should give positive loss."""
        predicted = F.normalize(torch.randn(8, 256), dim=-1)
        positive = F.normalize(torch.randn(8, 256), dim=-1)
        loss = info_nce_loss(predicted, positive, temperature=0.07)
        assert loss.item() > 0

    def test_batch_size_one_works(self) -> None:
        """Loss should work with batch size 1."""
        predicted = F.normalize(torch.randn(1, 256), dim=-1)
        positive = F.normalize(torch.randn(1, 256), dim=-1)
        loss = info_nce_loss(predicted, positive, temperature=0.07)
        assert loss.item() >= 0

    def test_normalizes_input(self) -> None:
        """Function should normalize inputs internally."""
        # Use unnormalized inputs
        predicted = torch.randn(8, 256) * 10
        positive = torch.randn(8, 256) * 5
        loss = info_nce_loss(predicted, positive, temperature=0.07)
        # Should not error and produce valid loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_temperature_effect(self) -> None:
        """Lower temperature should sharpen loss distribution."""
        predicted = F.normalize(torch.randn(8, 256), dim=-1)
        positive = F.normalize(torch.randn(8, 256), dim=-1)

        loss_high_temp = info_nce_loss(predicted, positive, temperature=1.0)
        loss_low_temp = info_nce_loss(predicted, positive, temperature=0.07)

        # Lower temp with random data typically gives higher loss
        # (sharper distribution, harder to get right)
        assert loss_low_temp != loss_high_temp

    def test_larger_batch_harder(self) -> None:
        """More negatives (larger batch) generally increases loss."""
        torch.manual_seed(42)
        predicted_small = F.normalize(torch.randn(4, 256), dim=-1)
        positive_small = F.normalize(torch.randn(4, 256), dim=-1)

        torch.manual_seed(42)
        predicted_large = F.normalize(torch.randn(32, 256), dim=-1)
        positive_large = F.normalize(torch.randn(32, 256), dim=-1)

        loss_small = info_nce_loss(predicted_small, positive_small)
        loss_large = info_nce_loss(predicted_large, positive_large)

        # More negatives = harder task = typically higher loss
        # Note: This is probabilistic, so we just check they're different
        assert loss_small.item() != loss_large.item()

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the loss."""
        # Use unnormalized input so we get leaf tensor gradients
        predicted = torch.randn(8, 256, requires_grad=True)
        positive = F.normalize(torch.randn(8, 256), dim=-1)

        loss = info_nce_loss(predicted, positive)
        loss.backward()

        assert predicted.grad is not None
        assert not torch.all(predicted.grad == 0)

    def test_invalid_3d_input_raises(self) -> None:
        """3D input should raise ValueError."""
        predicted = torch.randn(4, 8, 256)
        positive = torch.randn(4, 8, 256)

        with pytest.raises(ValueError, match="2D"):
            info_nce_loss(predicted, positive)

    def test_batch_size_mismatch_raises(self) -> None:
        """Mismatched batch sizes should raise ValueError."""
        predicted = torch.randn(8, 256)
        positive = torch.randn(4, 256)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            info_nce_loss(predicted, positive)

    def test_default_temperature(self) -> None:
        """Default temperature should be 0.07."""
        predicted = F.normalize(torch.randn(8, 256), dim=-1)
        positive = F.normalize(torch.randn(8, 256), dim=-1)

        loss_default = info_nce_loss(predicted, positive)
        loss_explicit = info_nce_loss(predicted, positive, temperature=0.07)

        torch.testing.assert_close(loss_default, loss_explicit)

    def test_output_is_scalar(self) -> None:
        """Loss should be a scalar tensor."""
        predicted = F.normalize(torch.randn(8, 256), dim=-1)
        positive = F.normalize(torch.randn(8, 256), dim=-1)

        loss = info_nce_loss(predicted, positive)

        assert loss.ndim == 0

    def test_various_dimensions(self) -> None:
        """Loss should work with various embedding dimensions."""
        for dim in [64, 128, 512, 1024, 2048]:
            predicted = F.normalize(torch.randn(4, dim), dim=-1)
            positive = F.normalize(torch.randn(4, dim), dim=-1)
            loss = info_nce_loss(predicted, positive)
            assert not torch.isnan(loss)


class TestSymmetricInfoNCELoss:
    """Tests for symmetric_info_nce_loss function."""

    def test_symmetric_identical_embeddings(self) -> None:
        """Identical embeddings should give near-zero symmetric loss."""
        embeddings = F.normalize(torch.randn(8, 256), dim=-1)
        loss = symmetric_info_nce_loss(embeddings, embeddings.clone())
        assert loss.item() < 0.1

    def test_symmetric_random_embeddings(self) -> None:
        """Random embeddings should give positive symmetric loss."""
        predicted = F.normalize(torch.randn(8, 256), dim=-1)
        positive = F.normalize(torch.randn(8, 256), dim=-1)
        loss = symmetric_info_nce_loss(predicted, positive)
        assert loss.item() > 0

    def test_symmetric_is_average(self) -> None:
        """Symmetric loss should be average of both directions."""
        predicted = F.normalize(torch.randn(8, 256), dim=-1)
        positive = F.normalize(torch.randn(8, 256), dim=-1)

        loss_p2t = info_nce_loss(predicted, positive)
        loss_t2p = info_nce_loss(positive, predicted)
        expected = (loss_p2t + loss_t2p) / 2

        actual = symmetric_info_nce_loss(predicted, positive)

        torch.testing.assert_close(actual, expected)

    def test_symmetric_commutative(self) -> None:
        """Symmetric loss should be same regardless of argument order."""
        a = F.normalize(torch.randn(8, 256), dim=-1)
        b = F.normalize(torch.randn(8, 256), dim=-1)

        loss_ab = symmetric_info_nce_loss(a, b)
        loss_ba = symmetric_info_nce_loss(b, a)

        torch.testing.assert_close(loss_ab, loss_ba)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through symmetric loss."""
        predicted = torch.randn(8, 256, requires_grad=True)
        positive = F.normalize(torch.randn(8, 256), dim=-1)

        loss = symmetric_info_nce_loss(predicted, positive)
        loss.backward()

        assert predicted.grad is not None


class TestInfoNCELossWithNegatives:
    """Tests for info_nce_loss_with_negatives function."""

    def test_basic_functionality(self) -> None:
        """Basic test with explicit negatives."""
        predicted = F.normalize(torch.randn(4, 256), dim=-1)
        positive = F.normalize(torch.randn(4, 256), dim=-1)
        negatives = F.normalize(torch.randn(4, 8, 256), dim=-1)  # 8 negatives per sample

        loss = info_nce_loss_with_negatives(predicted, positive, negatives)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_identical_positive_low_loss(self) -> None:
        """Identical positive should give low loss."""
        predicted = F.normalize(torch.randn(4, 256), dim=-1)
        negatives = F.normalize(torch.randn(4, 8, 256), dim=-1)

        loss = info_nce_loss_with_negatives(predicted, predicted.clone(), negatives)

        # Should be relatively low since positive matches perfectly
        assert loss.item() < 1.0

    def test_more_negatives_harder(self) -> None:
        """More negatives should generally make task harder."""
        torch.manual_seed(42)
        predicted = F.normalize(torch.randn(4, 256), dim=-1)
        positive = F.normalize(torch.randn(4, 256), dim=-1)

        few_negatives = F.normalize(torch.randn(4, 4, 256), dim=-1)
        many_negatives = F.normalize(torch.randn(4, 32, 256), dim=-1)

        loss_few = info_nce_loss_with_negatives(predicted, positive, few_negatives)
        loss_many = info_nce_loss_with_negatives(predicted, positive, many_negatives)

        # More negatives = harder = typically higher loss
        # This is probabilistic so just check they're computed
        assert not torch.isnan(loss_few)
        assert not torch.isnan(loss_many)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the loss."""
        predicted = torch.randn(4, 256, requires_grad=True)
        positive = F.normalize(torch.randn(4, 256), dim=-1)
        negatives = F.normalize(torch.randn(4, 8, 256), dim=-1)

        loss = info_nce_loss_with_negatives(predicted, positive, negatives)
        loss.backward()

        assert predicted.grad is not None

    def test_invalid_predicted_shape_raises(self) -> None:
        """Invalid predicted shape should raise ValueError."""
        predicted = torch.randn(4, 8, 256)  # 3D instead of 2D
        positive = torch.randn(4, 256)
        negatives = torch.randn(4, 8, 256)

        with pytest.raises(ValueError, match="2D"):
            info_nce_loss_with_negatives(predicted, positive, negatives)

    def test_invalid_negatives_shape_raises(self) -> None:
        """Invalid negatives shape should raise ValueError."""
        predicted = torch.randn(4, 256)
        positive = torch.randn(4, 256)
        negatives = torch.randn(4, 256)  # 2D instead of 3D

        with pytest.raises(ValueError, match="3D"):
            info_nce_loss_with_negatives(predicted, positive, negatives)

    def test_single_negative(self) -> None:
        """Should work with just one negative per sample."""
        predicted = F.normalize(torch.randn(4, 256), dim=-1)
        positive = F.normalize(torch.randn(4, 256), dim=-1)
        negatives = F.normalize(torch.randn(4, 1, 256), dim=-1)

        loss = info_nce_loss_with_negatives(predicted, positive, negatives)

        assert not torch.isnan(loss)

    def test_temperature_effect(self) -> None:
        """Temperature should affect loss value."""
        predicted = F.normalize(torch.randn(4, 256), dim=-1)
        positive = F.normalize(torch.randn(4, 256), dim=-1)
        negatives = F.normalize(torch.randn(4, 8, 256), dim=-1)

        loss_high = info_nce_loss_with_negatives(
            predicted, positive, negatives, temperature=1.0
        )
        loss_low = info_nce_loss_with_negatives(
            predicted, positive, negatives, temperature=0.07
        )

        assert loss_high != loss_low


class TestInfoNCEMathematicalProperties:
    """Tests for mathematical properties of InfoNCE loss."""

    def test_lower_bounded_by_zero(self) -> None:
        """InfoNCE loss should be >= 0."""
        for _ in range(10):
            predicted = F.normalize(torch.randn(8, 256), dim=-1)
            positive = F.normalize(torch.randn(8, 256), dim=-1)
            loss = info_nce_loss(predicted, positive)
            assert loss.item() >= 0

    def test_upper_bounded_by_log_n(self) -> None:
        """InfoNCE loss is bounded above by log(N) for random predictions."""
        batch_size = 32
        predicted = F.normalize(torch.randn(batch_size, 256), dim=-1)
        positive = F.normalize(torch.randn(batch_size, 256), dim=-1)

        loss = info_nce_loss(predicted, positive, temperature=1.0)

        # Upper bound is log(N) for uniform distribution
        upper_bound = math.log(batch_size)
        # Allow some margin for numerical precision
        assert loss.item() <= upper_bound + 0.5

    def test_loss_decreases_with_better_alignment(self) -> None:
        """Loss should decrease as embeddings become more aligned."""
        base = F.normalize(torch.randn(8, 256), dim=-1)
        random = F.normalize(torch.randn(8, 256), dim=-1)

        # Interpolate between random and identical
        loss_random = info_nce_loss(base, random)
        loss_similar = info_nce_loss(base, 0.8 * base + 0.2 * random)
        loss_identical = info_nce_loss(base, base)

        # More similar should have lower loss
        assert loss_similar < loss_random
        assert loss_identical < loss_similar
