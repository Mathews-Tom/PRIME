"""Unit tests for predictor type definitions."""

from __future__ import annotations

import numpy as np
import pytest

from prime.predictor import (
    CheckpointMetadata,
    PredictorInput,
    PredictorOutput,
    TrainingBatch,
    TrainingMetrics,
)


class TestPredictorInput:
    """Tests for PredictorInput data class."""

    def test_valid_input(self) -> None:
        """Test valid predictor input creation."""
        context = np.random.randn(5, 1024).astype(np.float32)
        query = np.random.randn(1024).astype(np.float32)

        pred_input = PredictorInput(
            context_embeddings=context,
            query_embedding=query,
        )
        assert pred_input.context_embeddings.shape == (5, 1024)
        assert pred_input.query_embedding.shape == (1024,)

    def test_invalid_context_shape(self) -> None:
        """Test that invalid context shape raises error."""
        # 1D context (should be 2D)
        context = np.random.randn(1024).astype(np.float32)
        query = np.random.randn(1024).astype(np.float32)

        with pytest.raises(ValueError, match="context_embeddings must be 2D"):
            PredictorInput(context_embeddings=context, query_embedding=query)

    def test_invalid_query_shape(self) -> None:
        """Test that invalid query shape raises error."""
        # 2D query (should be 1D)
        context = np.random.randn(5, 1024).astype(np.float32)
        query = np.random.randn(1, 1024).astype(np.float32)

        with pytest.raises(ValueError, match="query_embedding must be 1D"):
            PredictorInput(context_embeddings=context, query_embedding=query)

    def test_dimension_mismatch(self) -> None:
        """Test that dimension mismatch raises error."""
        context = np.random.randn(5, 1024).astype(np.float32)
        query = np.random.randn(768).astype(np.float32)  # Different dimension

        with pytest.raises(ValueError, match="Dimension mismatch"):
            PredictorInput(context_embeddings=context, query_embedding=query)

    def test_frozen(self) -> None:
        """Test that input is immutable."""
        context = np.random.randn(5, 1024).astype(np.float32)
        query = np.random.randn(1024).astype(np.float32)

        pred_input = PredictorInput(
            context_embeddings=context,
            query_embedding=query,
        )

        with pytest.raises(AttributeError):
            pred_input.context_embeddings = context  # type: ignore[misc]


class TestPredictorOutput:
    """Tests for PredictorOutput data class."""

    def test_valid_output(self) -> None:
        """Test valid predictor output creation."""
        embedding = np.random.randn(1024).astype(np.float32)

        output = PredictorOutput(
            predicted_embedding=embedding,
            confidence=0.95,
        )
        assert output.predicted_embedding.shape == (1024,)
        assert output.confidence == 0.95

    def test_default_confidence(self) -> None:
        """Test default confidence value."""
        embedding = np.random.randn(1024).astype(np.float32)

        output = PredictorOutput(predicted_embedding=embedding)
        assert output.confidence == 1.0

    def test_frozen(self) -> None:
        """Test that output is immutable."""
        embedding = np.random.randn(1024).astype(np.float32)

        output = PredictorOutput(predicted_embedding=embedding)

        with pytest.raises(AttributeError):
            output.confidence = 0.5  # type: ignore[misc]


class TestTrainingBatch:
    """Tests for TrainingBatch data class."""

    def test_valid_batch(self) -> None:
        """Test valid training batch creation."""
        batch_size = 32
        context_len = 5
        dim = 1024

        batch = TrainingBatch(
            context_embeddings=np.random.randn(batch_size, context_len, dim).astype(
                np.float32
            ),
            query_embeddings=np.random.randn(batch_size, dim).astype(np.float32),
            target_embeddings=np.random.randn(batch_size, dim).astype(np.float32),
        )
        assert batch.context_embeddings.shape == (32, 5, 1024)
        assert batch.query_embeddings.shape == (32, 1024)
        assert batch.target_embeddings.shape == (32, 1024)

    def test_frozen(self) -> None:
        """Test that batch is immutable."""
        batch = TrainingBatch(
            context_embeddings=np.random.randn(4, 5, 1024).astype(np.float32),
            query_embeddings=np.random.randn(4, 1024).astype(np.float32),
            target_embeddings=np.random.randn(4, 1024).astype(np.float32),
        )

        with pytest.raises(AttributeError):
            batch.context_embeddings = None  # type: ignore[misc]


class TestTrainingMetrics:
    """Tests for TrainingMetrics data class."""

    def test_valid_metrics(self) -> None:
        """Test valid training metrics creation."""
        metrics = TrainingMetrics(
            loss=1.5,
            accuracy=0.85,
            learning_rate=1e-4,
            gradient_norm=0.5,
        )
        assert metrics.loss == 1.5
        assert metrics.accuracy == 0.85
        assert metrics.learning_rate == 1e-4
        assert metrics.gradient_norm == 0.5

    def test_optional_gradient_norm(self) -> None:
        """Test optional gradient norm."""
        metrics = TrainingMetrics(
            loss=1.5,
            accuracy=0.85,
            learning_rate=1e-4,
        )
        assert metrics.gradient_norm is None

    def test_frozen(self) -> None:
        """Test that metrics are immutable."""
        metrics = TrainingMetrics(
            loss=1.5,
            accuracy=0.85,
            learning_rate=1e-4,
        )

        with pytest.raises(AttributeError):
            metrics.loss = 2.0  # type: ignore[misc]


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata data class."""

    def test_valid_metadata(self) -> None:
        """Test valid checkpoint metadata creation."""
        metadata = CheckpointMetadata(
            epoch=5,
            step=10000,
            loss=1.2,
            config_hash="abc123def456",
        )
        assert metadata.epoch == 5
        assert metadata.step == 10000
        assert metadata.loss == 1.2
        assert metadata.config_hash == "abc123def456"

    def test_frozen(self) -> None:
        """Test that metadata is immutable."""
        metadata = CheckpointMetadata(
            epoch=5,
            step=10000,
            loss=1.2,
            config_hash="abc123",
        )

        with pytest.raises(AttributeError):
            metadata.epoch = 10  # type: ignore[misc]
