"""Unit tests for EmbeddingPredictor module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from prime.predictor import (
    CheckpointError,
    EmbeddingPredictor,
    PredictorConfig,
    PredictorShapeError,
    PredictorTransformerBlock,
    create_predictor,
)


class TestPredictorTransformerBlock:
    """Tests for PredictorTransformerBlock."""

    def test_forward_shape(self) -> None:
        """Test that output shape matches input shape."""
        block = PredictorTransformerBlock(
            hidden_dim=256,
            num_heads=4,
            dropout=0.1,
        )
        x = torch.randn(2, 10, 256)
        output = block(x)
        assert output.shape == x.shape

    def test_different_hidden_dims(self) -> None:
        """Test with different hidden dimensions."""
        for hidden_dim in [128, 256, 512]:
            block = PredictorTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=0.0,
            )
            x = torch.randn(1, 5, hidden_dim)
            output = block(x)
            assert output.shape == (1, 5, hidden_dim)

    def test_pre_norm_architecture(self) -> None:
        """Test that pre-norm is applied (output differs from simple pass-through)."""
        block = PredictorTransformerBlock(
            hidden_dim=128,
            num_heads=4,
            dropout=0.0,
        )
        block.eval()
        x = torch.randn(1, 3, 128)
        output = block(x)
        # Pre-norm should produce different output
        assert not torch.allclose(output, x)


class TestEmbeddingPredictor:
    """Tests for EmbeddingPredictor."""

    @pytest.fixture
    def small_config(self) -> PredictorConfig:
        """Small config for fast testing."""
        return PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
            max_context_length=5,
            max_batch_size=8,
        )

    @pytest.fixture
    def predictor(self, small_config: PredictorConfig) -> EmbeddingPredictor:
        """Create predictor with small config."""
        return EmbeddingPredictor(small_config)

    def test_forward_output_shape(self, predictor: EmbeddingPredictor) -> None:
        """Test that forward produces correct output shape."""
        batch_size = 2
        context_len = 3
        input_dim = 128

        context = torch.randn(batch_size, context_len, input_dim)
        query = torch.randn(batch_size, input_dim)

        output = predictor(context, query)
        assert output.shape == (batch_size, 128)

    def test_output_normalized(self, predictor: EmbeddingPredictor) -> None:
        """Test that output is L2-normalized."""
        context = torch.randn(4, 3, 128)
        query = torch.randn(4, 128)

        output = predictor(context, query)
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_batch_size_one(self, predictor: EmbeddingPredictor) -> None:
        """Test with single sample batch."""
        context = torch.randn(1, 3, 128)
        query = torch.randn(1, 128)

        output = predictor(context, query)
        assert output.shape == (1, 128)

    def test_single_context(self, predictor: EmbeddingPredictor) -> None:
        """Test with single context embedding."""
        context = torch.randn(2, 1, 128)
        query = torch.randn(2, 128)

        output = predictor(context, query)
        assert output.shape == (2, 128)

    def test_max_context_length(self, predictor: EmbeddingPredictor) -> None:
        """Test with maximum context length."""
        context = torch.randn(2, 5, 128)  # max_context_length=5
        query = torch.randn(2, 128)

        output = predictor(context, query)
        assert output.shape == (2, 128)

    def test_deterministic_output(self, predictor: EmbeddingPredictor) -> None:
        """Test that same input produces same output in eval mode."""
        predictor.eval()
        context = torch.randn(2, 3, 128)
        query = torch.randn(2, 128)

        output1 = predictor(context, query)
        output2 = predictor(context, query)
        assert torch.allclose(output1, output2)


class TestPredictorShapeValidation:
    """Tests for shape validation in EmbeddingPredictor."""

    @pytest.fixture
    def config(self) -> PredictorConfig:
        """Config for testing."""
        return PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
            max_context_length=5,
            max_batch_size=8,
        )

    @pytest.fixture
    def predictor(self, config: PredictorConfig) -> EmbeddingPredictor:
        """Create predictor."""
        return EmbeddingPredictor(config)

    def test_invalid_context_ndim(self, predictor: EmbeddingPredictor) -> None:
        """Test error on invalid context dimensions."""
        context = torch.randn(128)  # 1D instead of 3D
        query = torch.randn(1, 128)

        with pytest.raises(PredictorShapeError, match="must be 3D"):
            predictor(context, query)

    def test_invalid_query_ndim(self, predictor: EmbeddingPredictor) -> None:
        """Test error on invalid query dimensions."""
        context = torch.randn(1, 3, 128)
        query = torch.randn(128)  # 1D instead of 2D

        with pytest.raises(PredictorShapeError, match="must be 2D"):
            predictor(context, query)

    def test_dimension_mismatch(self, predictor: EmbeddingPredictor) -> None:
        """Test error on dimension mismatch."""
        context = torch.randn(1, 3, 256)  # 256 != config.input_dim (128)
        query = torch.randn(1, 128)

        with pytest.raises(PredictorShapeError, match="!= config.input_dim"):
            predictor(context, query)

    def test_batch_size_mismatch(self, predictor: EmbeddingPredictor) -> None:
        """Test error on batch size mismatch."""
        context = torch.randn(2, 3, 128)
        query = torch.randn(3, 128)  # Different batch size

        with pytest.raises(PredictorShapeError, match="Batch size mismatch"):
            predictor(context, query)

    def test_context_too_long(self, predictor: EmbeddingPredictor) -> None:
        """Test error when context exceeds max length."""
        context = torch.randn(1, 10, 128)  # 10 > max_context_length (5)
        query = torch.randn(1, 128)

        with pytest.raises(PredictorShapeError, match="max_context_length"):
            predictor(context, query)

    def test_batch_too_large(self, predictor: EmbeddingPredictor) -> None:
        """Test error when batch exceeds max size."""
        context = torch.randn(16, 3, 128)  # 16 > max_batch_size (8)
        query = torch.randn(16, 128)

        with pytest.raises(PredictorShapeError, match="max_batch_size"):
            predictor(context, query)


class TestPredictorPredict:
    """Tests for predict() numpy interface."""

    @pytest.fixture
    def predictor(self) -> EmbeddingPredictor:
        """Create predictor."""
        config = PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
            max_context_length=5,
            max_batch_size=8,
        )
        return EmbeddingPredictor(config)

    def test_unbatched_input(self, predictor: EmbeddingPredictor) -> None:
        """Test predict with unbatched input."""
        context = np.random.randn(3, 128).astype(np.float32)
        query = np.random.randn(128).astype(np.float32)

        output = predictor.predict(context, query)
        assert output.shape == (128,)

    def test_batched_input(self, predictor: EmbeddingPredictor) -> None:
        """Test predict with batched input."""
        context = np.random.randn(4, 3, 128).astype(np.float32)
        query = np.random.randn(4, 128).astype(np.float32)

        output = predictor.predict(context, query)
        assert output.shape == (4, 128)

    def test_output_normalized(self, predictor: EmbeddingPredictor) -> None:
        """Test that predict output is normalized."""
        context = np.random.randn(3, 128).astype(np.float32)
        query = np.random.randn(128).astype(np.float32)

        output = predictor.predict(context, query)
        norm = np.linalg.norm(output)
        assert np.isclose(norm, 1.0, atol=1e-5)


class TestPredictorCheckpoint:
    """Tests for checkpoint save/load."""

    @pytest.fixture
    def predictor(self) -> EmbeddingPredictor:
        """Create predictor."""
        config = PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
        )
        return EmbeddingPredictor(config)

    def test_save_and_load(self, predictor: EmbeddingPredictor) -> None:
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            predictor.save_checkpoint(path, epoch=5, step=1000, loss=1.5)
            assert path.exists()

            # Load into new predictor
            config = predictor.config
            new_predictor = EmbeddingPredictor(config)
            metadata = new_predictor.load_checkpoint(path)

            assert metadata.epoch == 5
            assert metadata.step == 1000
            assert metadata.loss == 1.5

    def test_load_preserves_weights(self, predictor: EmbeddingPredictor) -> None:
        """Test that loading checkpoint preserves weights."""
        predictor.eval()
        context = torch.randn(1, 3, 128)
        query = torch.randn(1, 128)

        output_before = predictor(context, query).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            predictor.save_checkpoint(path)

            new_predictor = EmbeddingPredictor(predictor.config)
            new_predictor.load_checkpoint(path)
            new_predictor.eval()

            output_after = new_predictor(context, query)
            assert torch.allclose(output_before, output_after)

    def test_load_nonexistent_file(self, predictor: EmbeddingPredictor) -> None:
        """Test error on loading nonexistent file."""
        with pytest.raises(CheckpointError, match="not found"):
            predictor.load_checkpoint("/nonexistent/path.pt")

    def test_config_mismatch(self, predictor: EmbeddingPredictor) -> None:
        """Test error on config mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            predictor.save_checkpoint(path)

            # Create predictor with different config
            different_config = PredictorConfig(
                input_dim=128,
                hidden_dim=512,  # Different
                output_dim=128,
                num_layers=2,
                num_heads=8,
            )
            different_predictor = EmbeddingPredictor(different_config)

            with pytest.raises(CheckpointError, match="Config mismatch"):
                different_predictor.load_checkpoint(path)


class TestPredictorProperties:
    """Tests for predictor properties."""

    def test_num_parameters(self) -> None:
        """Test parameter counting."""
        config = PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
        )
        predictor = EmbeddingPredictor(config)

        assert predictor.num_parameters > 0
        assert predictor.num_trainable_parameters == predictor.num_parameters

    def test_attention_head_dim(self) -> None:
        """Test attention head dimension calculation."""
        config = PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
        )
        predictor = EmbeddingPredictor(config)

        assert predictor.get_attention_head_dim() == 64  # 256 / 4


class TestONNXExport:
    """Tests for ONNX export functionality."""

    @pytest.fixture
    def predictor(self) -> EmbeddingPredictor:
        """Create predictor for ONNX export tests."""
        config = PredictorConfig(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
            num_heads=4,
            max_context_length=8,
        )
        return EmbeddingPredictor(config)

    def test_export_creates_file(self, predictor: EmbeddingPredictor) -> None:
        """Test that export_onnx creates ONNX file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            predictor.export_onnx(path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_export_creates_parent_dirs(
        self, predictor: EmbeddingPredictor
    ) -> None:
        """Test that export creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "model.onnx"
            predictor.export_onnx(path)
            assert path.exists()

    def test_export_preserves_training_mode(
        self, predictor: EmbeddingPredictor
    ) -> None:
        """Test that export restores training mode after completion."""
        predictor.train()
        assert predictor.training

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            predictor.export_onnx(path)

        assert predictor.training

    def test_export_from_eval_mode(self, predictor: EmbeddingPredictor) -> None:
        """Test export works from eval mode."""
        predictor.eval()
        assert not predictor.training

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            predictor.export_onnx(path)
            assert path.exists()

        assert not predictor.training

    def test_export_with_custom_opset(
        self, predictor: EmbeddingPredictor
    ) -> None:
        """Test export with custom opset version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            predictor.export_onnx(path, opset_version=14)
            assert path.exists()

    def test_export_string_path(self, predictor: EmbeddingPredictor) -> None:
        """Test export accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.onnx")
            predictor.export_onnx(path)
            assert Path(path).exists()


class TestCreatePredictor:
    """Tests for create_predictor factory function."""

    def test_default_config(self) -> None:
        """Test creating predictor with default config."""
        predictor = create_predictor()
        assert predictor.config.input_dim == 1024
        assert predictor.config.hidden_dim == 2048

    def test_custom_config(self) -> None:
        """Test creating predictor with custom config."""
        config = PredictorConfig(
            input_dim=512,
            hidden_dim=1024,
            output_dim=512,
        )
        predictor = create_predictor(config)
        assert predictor.config.input_dim == 512
        assert predictor.config.hidden_dim == 1024
