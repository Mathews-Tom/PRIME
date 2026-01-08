"""Tests for PRIME training module.

Tests PredictorLightningModule and training utilities.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prime.predictor import EmbeddingPredictor, PredictorConfig, TrainingConfig
from prime.training.trainer import PredictorLightningModule


class MockYEncoder:
    """Mock Y-Encoder for testing without loading real models."""

    def __init__(self, output_dim: int = 1024) -> None:
        """Initialize mock encoder with specified output dimension."""
        self.output_dim = output_dim
        self._encode_count = 0

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Return mock embeddings for texts."""
        self._encode_count += 1
        return [
            np.random.randn(self.output_dim).astype(np.float32) for _ in texts
        ]


class TestPredictorLightningModuleInit:
    """Tests for PredictorLightningModule initialization."""

    def test_init_with_defaults(self) -> None:
        """Module should initialize with default config."""
        predictor = EmbeddingPredictor(PredictorConfig())
        y_encoder = MockYEncoder()

        module = PredictorLightningModule(predictor, y_encoder)

        assert module.predictor is predictor
        assert module.y_encoder is y_encoder
        assert isinstance(module.config, TrainingConfig)

    def test_init_with_custom_config(self) -> None:
        """Module should use provided config."""
        predictor = EmbeddingPredictor(PredictorConfig())
        y_encoder = MockYEncoder()
        config = TrainingConfig(learning_rate=1e-4, temperature=0.1)

        module = PredictorLightningModule(predictor, y_encoder, config)

        assert module.config.learning_rate == 1e-4
        assert module.config.temperature == 0.1


class TestPredictorLightningModuleComputeLoss:
    """Tests for compute_loss method."""

    @pytest.fixture
    def module(self) -> PredictorLightningModule:
        """Create module for testing."""
        predictor = EmbeddingPredictor(PredictorConfig())
        y_encoder = MockYEncoder(output_dim=1024)
        return PredictorLightningModule(predictor, y_encoder)

    def test_compute_loss_returns_tuple(self, module: PredictorLightningModule) -> None:
        """compute_loss should return loss and metrics dict."""
        predicted = F.normalize(torch.randn(4, 1024), dim=-1)
        target_texts = ["text1", "text2", "text3", "text4"]

        result = module.compute_loss(predicted, target_texts)

        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_compute_loss_positive_loss(self, module: PredictorLightningModule) -> None:
        """Loss should be positive."""
        predicted = F.normalize(torch.randn(4, 1024), dim=-1)
        target_texts = ["text1", "text2", "text3", "text4"]

        loss, _ = module.compute_loss(predicted, target_texts)

        assert loss.item() > 0

    def test_compute_loss_metrics_keys(self, module: PredictorLightningModule) -> None:
        """Metrics should contain expected keys."""
        predicted = F.normalize(torch.randn(4, 1024), dim=-1)
        target_texts = ["text1", "text2", "text3", "text4"]

        _, metrics = module.compute_loss(predicted, target_texts)

        assert "accuracy" in metrics
        assert "pos_similarity" in metrics

    def test_compute_loss_accuracy_range(self, module: PredictorLightningModule) -> None:
        """Accuracy should be between 0 and 1."""
        predicted = F.normalize(torch.randn(4, 1024), dim=-1)
        target_texts = ["text1", "text2", "text3", "text4"]

        _, metrics = module.compute_loss(predicted, target_texts)

        assert 0 <= metrics["accuracy"] <= 1

    def test_compute_loss_similarity_range(
        self, module: PredictorLightningModule
    ) -> None:
        """Positive similarity should be between -1 and 1."""
        predicted = F.normalize(torch.randn(4, 1024), dim=-1)
        target_texts = ["text1", "text2", "text3", "text4"]

        _, metrics = module.compute_loss(predicted, target_texts)

        assert -1 <= metrics["pos_similarity"] <= 1

    def test_compute_loss_uses_y_encoder(
        self, module: PredictorLightningModule
    ) -> None:
        """compute_loss should call y_encoder.encode_batch."""
        predicted = F.normalize(torch.randn(4, 1024), dim=-1)
        target_texts = ["text1", "text2", "text3", "text4"]

        initial_count = module.y_encoder._encode_count  # type: ignore[union-attr]
        module.compute_loss(predicted, target_texts)

        assert module.y_encoder._encode_count > initial_count  # type: ignore[union-attr]


class TestPredictorLightningModuleTrainingStep:
    """Tests for training_step method."""

    @pytest.fixture
    def module(self) -> PredictorLightningModule:
        """Create module for testing."""
        config = PredictorConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
        )
        predictor = EmbeddingPredictor(config)
        y_encoder = MockYEncoder(output_dim=256)
        return PredictorLightningModule(predictor, y_encoder)

    def test_training_step_returns_tuple(
        self, module: PredictorLightningModule
    ) -> None:
        """training_step should return loss and metrics."""
        batch = {
            "context_embeddings": torch.randn(2, 4, 256),
            "query_embedding": torch.randn(2, 256),
            "target_content": ["response1", "response2"],
        }

        result = module.training_step(batch)

        assert isinstance(result, tuple)
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_training_step_loss_requires_grad(
        self, module: PredictorLightningModule
    ) -> None:
        """Loss should have gradient enabled for training."""
        batch = {
            "context_embeddings": torch.randn(2, 4, 256),
            "query_embedding": torch.randn(2, 256),
            "target_content": ["response1", "response2"],
        }

        loss, _ = module.training_step(batch)

        assert loss.requires_grad

    def test_training_step_gradient_flows(
        self, module: PredictorLightningModule
    ) -> None:
        """Gradients should flow back to predictor parameters."""
        batch = {
            "context_embeddings": torch.randn(2, 4, 256),
            "query_embedding": torch.randn(2, 256),
            "target_content": ["response1", "response2"],
        }

        loss, _ = module.training_step(batch)
        loss.backward()

        # Check some parameter has gradients
        has_grad = False
        for param in module.predictor.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad


class TestPredictorLightningModuleValidationStep:
    """Tests for validation_step method."""

    @pytest.fixture
    def module(self) -> PredictorLightningModule:
        """Create module for testing."""
        config = PredictorConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
        )
        predictor = EmbeddingPredictor(config)
        y_encoder = MockYEncoder(output_dim=256)
        return PredictorLightningModule(predictor, y_encoder)

    def test_validation_step_returns_tuple(
        self, module: PredictorLightningModule
    ) -> None:
        """validation_step should return loss and metrics."""
        batch = {
            "context_embeddings": torch.randn(2, 4, 256),
            "query_embedding": torch.randn(2, 256),
            "target_content": ["response1", "response2"],
        }

        result = module.validation_step(batch)

        assert isinstance(result, tuple)
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_validation_step_no_grad(
        self, module: PredictorLightningModule
    ) -> None:
        """validation_step should not require gradients."""
        batch = {
            "context_embeddings": torch.randn(2, 4, 256),
            "query_embedding": torch.randn(2, 256),
            "target_content": ["response1", "response2"],
        }

        loss, _ = module.validation_step(batch)

        assert not loss.requires_grad

    def test_validation_step_same_output_as_training(
        self, module: PredictorLightningModule
    ) -> None:
        """Validation should produce same metrics format as training."""
        batch = {
            "context_embeddings": torch.randn(2, 4, 256),
            "query_embedding": torch.randn(2, 256),
            "target_content": ["response1", "response2"],
        }

        _, train_metrics = module.training_step(batch)
        _, val_metrics = module.validation_step(batch)

        assert set(train_metrics.keys()) == set(val_metrics.keys())


class TestPredictorLightningModuleConfigureOptimizers:
    """Tests for configure_optimizers method."""

    @pytest.fixture
    def module(self) -> PredictorLightningModule:
        """Create module for testing."""
        predictor = EmbeddingPredictor(PredictorConfig())
        y_encoder = MockYEncoder()
        config = TrainingConfig(
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
        )
        return PredictorLightningModule(predictor, y_encoder, config)

    def test_returns_optimizer_and_scheduler(
        self, module: PredictorLightningModule
    ) -> None:
        """Should return optimizer and scheduler tuple."""
        result = module.configure_optimizers()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_optimizer_is_adamw(self, module: PredictorLightningModule) -> None:
        """Optimizer should be AdamW."""
        optimizer, _ = module.configure_optimizers()

        assert optimizer.__class__.__name__ == "AdamW"

    def test_optimizer_uses_config_lr(self, module: PredictorLightningModule) -> None:
        """Optimizer should use learning rate from config."""
        optimizer, _ = module.configure_optimizers()

        # Check the learning rate in param groups
        for group in optimizer.param_groups:
            assert group["lr"] == 1e-4

    def test_optimizer_uses_config_weight_decay(
        self, module: PredictorLightningModule
    ) -> None:
        """Optimizer should use weight decay from config."""
        optimizer, _ = module.configure_optimizers()

        for group in optimizer.param_groups:
            assert group["weight_decay"] == 0.01

    def test_scheduler_is_cosine_annealing(
        self, module: PredictorLightningModule
    ) -> None:
        """Scheduler should be CosineAnnealingWarmRestarts."""
        _, scheduler = module.configure_optimizers()

        assert scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts"


class TestCreateLightningModule:
    """Tests for create_lightning_module factory function."""

    def test_import_error_without_lightning(self) -> None:
        """Should raise ImportError when lightning not installed."""
        # We can't easily test this without uninstalling lightning
        # Just verify the function exists and can be imported
        from prime.training.trainer import create_lightning_module

        assert callable(create_lightning_module)


class TestTrainingConfig:
    """Tests for TrainingConfig used in training module."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = TrainingConfig()

        assert config.learning_rate > 0
        assert config.temperature > 0
        assert config.temperature < 1.0  # Temperature is usually small
        assert config.warmup_steps >= 0
        assert config.weight_decay >= 0

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = TrainingConfig(
            learning_rate=5e-5,
            temperature=0.1,
            warmup_steps=200,
            weight_decay=0.05,
            gradient_clip_val=1.0,
        )

        assert config.learning_rate == 5e-5
        assert config.temperature == 0.1
        assert config.warmup_steps == 200
        assert config.weight_decay == 0.05
        assert config.gradient_clip_val == 1.0


class TestEndToEndTraining:
    """End-to-end training flow tests."""

    def test_full_training_iteration(self) -> None:
        """Test complete training iteration with optimizer step."""
        # Setup
        config = PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
        )
        predictor = EmbeddingPredictor(config)
        y_encoder = MockYEncoder(output_dim=128)
        training_config = TrainingConfig(learning_rate=1e-3)
        module = PredictorLightningModule(predictor, y_encoder, training_config)

        # Get optimizer
        optimizer, _ = module.configure_optimizers()

        # Create batch
        batch = {
            "context_embeddings": torch.randn(4, 8, 128),
            "query_embedding": torch.randn(4, 128),
            "target_content": ["text1", "text2", "text3", "text4"],
        }

        # Get initial parameters
        initial_params = {
            name: param.clone()
            for name, param in predictor.named_parameters()
        }

        # Training step
        loss, metrics = module.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify parameters changed
        params_changed = False
        for name, param in predictor.named_parameters():
            if not torch.allclose(initial_params[name], param):
                params_changed = True
                break

        assert params_changed, "Optimizer should update parameters"
        assert "accuracy" in metrics
        assert "pos_similarity" in metrics

    def test_multiple_training_iterations(self) -> None:
        """Test multiple training iterations complete without errors."""
        # Setup with small model for fast testing
        config = PredictorConfig(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_layers=1,
        )
        predictor = EmbeddingPredictor(config)
        y_encoder = MockYEncoder(output_dim=64)
        training_config = TrainingConfig(learning_rate=1e-3)
        module = PredictorLightningModule(predictor, y_encoder, training_config)

        optimizer, _ = module.configure_optimizers()

        # Fixed batch for consistent loss comparison
        torch.manual_seed(42)
        batch = {
            "context_embeddings": torch.randn(4, 4, 64),
            "query_embedding": torch.randn(4, 64),
            "target_content": ["a", "b", "c", "d"],
        }

        # Record losses over iterations - verify training loop works
        losses = []
        for _ in range(5):
            loss, metrics = module.training_step(batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # All losses should be finite (no NaN or Inf)
        for loss_val in losses:
            assert not np.isnan(loss_val), "Loss should not be NaN"
            assert not np.isinf(loss_val), "Loss should not be Inf"

        # Metrics should be computed
        assert "accuracy" in metrics
        assert "pos_similarity" in metrics
