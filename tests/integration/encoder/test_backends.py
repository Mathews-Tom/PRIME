"""Integration tests for encoder backends.

These tests validate that all supported embedding models work correctly.
They require model downloads and are marked as slow/integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from prime.encoder import (
    BGE_LARGE_CONFIG,
    MINILM_CONFIG,
    YEncoder,
    YEncoderConfig,
)


@pytest.mark.integration
class TestMiniLMBackend:
    """Tests for MiniLM encoder backend."""

    @pytest.fixture(scope="class")
    def encoder(self) -> YEncoder:
        """MiniLM encoder instance."""
        return YEncoder(MINILM_CONFIG)

    def test_minilm_produces_correct_dim(self, encoder: YEncoder) -> None:
        """MiniLM produces 384-dimensional embeddings."""
        embedding = encoder.encode("Test sentence")
        assert embedding.shape == (384,)

    def test_minilm_normalized_output(self, encoder: YEncoder) -> None:
        """MiniLM output is L2-normalized."""
        embedding = encoder.encode("Test sentence")
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_minilm_batch_encoding(self, encoder: YEncoder) -> None:
        """MiniLM handles batch encoding correctly."""
        texts = ["First", "Second", "Third"]
        embeddings = encoder.encode_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (384,)

    def test_minilm_semantic_quality(self, encoder: YEncoder) -> None:
        """MiniLM captures semantic similarity."""
        emb1 = encoder.encode("The cat sat on the mat")
        emb2 = encoder.encode("A feline rested on the rug")
        emb3 = encoder.encode("Python programming language")

        # Similar texts should have higher cosine similarity
        sim_similar = np.dot(emb1, emb2)
        sim_different = np.dot(emb1, emb3)
        assert sim_similar > sim_different


@pytest.mark.integration
@pytest.mark.slow
class TestBGELargeBackend:
    """Tests for BGE Large encoder backend."""

    @pytest.fixture(scope="class")
    def encoder(self) -> YEncoder:
        """BGE Large encoder instance."""
        return YEncoder(BGE_LARGE_CONFIG)

    def test_bge_produces_correct_dim(self, encoder: YEncoder) -> None:
        """BGE Large produces 1024-dimensional embeddings."""
        embedding = encoder.encode("Test sentence")
        assert embedding.shape == (1024,)

    def test_bge_normalized_output(self, encoder: YEncoder) -> None:
        """BGE Large output is L2-normalized."""
        embedding = encoder.encode("Test sentence")
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_bge_uses_cls_pooling(self, encoder: YEncoder) -> None:
        """BGE Large uses CLS pooling as configured."""
        info = encoder.get_model_info()
        assert info["pooling_mode"] == "cls"

    def test_bge_semantic_quality(self, encoder: YEncoder) -> None:
        """BGE Large captures semantic similarity."""
        emb1 = encoder.encode("The cat sat on the mat")
        emb2 = encoder.encode("A feline rested on the rug")
        emb3 = encoder.encode("Python programming language")

        sim_similar = np.dot(emb1, emb2)
        sim_different = np.dot(emb1, emb3)
        assert sim_similar > sim_different


@pytest.mark.integration
@pytest.mark.slow
class TestModelSwitching:
    """Tests for runtime model switching."""

    def test_switch_between_models(self) -> None:
        """Can switch between different encoder models."""
        # Start with MiniLM
        minilm_encoder = YEncoder(MINILM_CONFIG)
        minilm_emb = minilm_encoder.encode("Test")
        assert minilm_emb.shape == (384,)

        # Switch to BGE Large
        bge_encoder = YEncoder(BGE_LARGE_CONFIG)
        bge_emb = bge_encoder.encode("Test")
        assert bge_emb.shape == (1024,)

        # Embeddings should be different shapes
        assert minilm_emb.shape != bge_emb.shape

    def test_multiple_encoder_instances(self) -> None:
        """Multiple encoder instances can coexist."""
        encoder1 = YEncoder(MINILM_CONFIG)
        encoder2 = YEncoder(MINILM_CONFIG)

        emb1 = encoder1.encode("Same text")
        emb2 = encoder2.encode("Same text")

        # Same model should produce identical embeddings
        assert np.array_equal(emb1, emb2)


@pytest.mark.integration
class TestDimensionValidation:
    """Tests for dimension validation at load time."""

    def test_dimension_mismatch_raises_at_load(self) -> None:
        """Mismatched dimension raises ModelLoadError at load time."""
        from prime.encoder import ModelLoadError

        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=1024,  # Wrong: MiniLM produces 384
        )
        with pytest.raises(ModelLoadError, match="dimension mismatch"):
            YEncoder(config)

    def test_correct_dimension_loads_successfully(self) -> None:
        """Correct dimension specification loads without error."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
        )
        encoder = YEncoder(config)
        assert encoder.embedding_dim == 384


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name,expected_dim",
    [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ],
)
def test_backend_produces_correct_dim(model_name: str, expected_dim: int) -> None:
    """Parametrized test that each backend produces correct dimension."""
    config = YEncoderConfig(
        model_name=model_name,
        embedding_dim=expected_dim,
    )
    encoder = YEncoder(config)
    embedding = encoder.encode("Test sentence")
    assert embedding.shape == (expected_dim,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
