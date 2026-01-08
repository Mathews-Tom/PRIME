"""Comprehensive tests for YEncoder implementation."""

from __future__ import annotations

import numpy as np
import pytest

from prime.encoder import (
    Encoder,
    EncodingError,
    ModelLoadError,
    YEncoder,
    YEncoderConfig,
)


class TestProtocolCompliance:
    """Tests for Encoder protocol compliance."""

    def test_encoder_implements_protocol(self, encoder: YEncoder) -> None:
        """YEncoder implements the Encoder protocol."""
        assert isinstance(encoder, Encoder)

    def test_encoder_has_required_properties(self, encoder: YEncoder) -> None:
        """YEncoder has all required protocol properties."""
        assert hasattr(encoder, "embedding_dim")
        assert hasattr(encoder, "max_length")
        assert hasattr(encoder, "model_name")

    def test_encoder_has_required_methods(self, encoder: YEncoder) -> None:
        """YEncoder has all required protocol methods."""
        assert hasattr(encoder, "encode")
        assert hasattr(encoder, "encode_batch")
        assert hasattr(encoder, "get_model_info")
        assert callable(encoder.encode)
        assert callable(encoder.encode_batch)
        assert callable(encoder.get_model_info)


class TestProperties:
    """Tests for encoder properties."""

    def test_embedding_dim_property(
        self, encoder: YEncoder, encoder_config: YEncoderConfig
    ) -> None:
        """Embedding dimension property returns correct value."""
        assert encoder.embedding_dim == encoder_config.embedding_dim

    def test_max_length_property(
        self, encoder: YEncoder, encoder_config: YEncoderConfig
    ) -> None:
        """Max length property returns correct value."""
        assert encoder.max_length == encoder_config.max_length

    def test_model_name_property(
        self, encoder: YEncoder, encoder_config: YEncoderConfig
    ) -> None:
        """Model name property returns correct value."""
        assert encoder.model_name == encoder_config.model_name


class TestCoreFunctionality:
    """Tests for core encoding functionality."""

    def test_encode_returns_correct_dim(self, encoder: YEncoder) -> None:
        """Encode produces embedding of correct dimension."""
        embedding = encoder.encode("Test sentence")
        assert embedding.shape == (encoder.embedding_dim,)

    def test_encode_returns_numpy_array(self, encoder: YEncoder) -> None:
        """Encode returns numpy array with float32 dtype."""
        embedding = encoder.encode("Test sentence")
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32

    def test_output_normalized(self, encoder: YEncoder) -> None:
        """Output embedding is L2-normalized (unit vector)."""
        embedding = encoder.encode("Test sentence for normalization")
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_batch_encode_returns_list(self, encoder: YEncoder) -> None:
        """Batch encode returns list of embeddings."""
        texts = ["First sentence", "Second sentence"]
        embeddings = encoder.encode_batch(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2

    def test_batch_encode_correct_dims(self, encoder: YEncoder) -> None:
        """Each batch embedding has correct dimension."""
        texts = ["First", "Second", "Third"]
        embeddings = encoder.encode_batch(texts)
        for emb in embeddings:
            assert emb.shape == (encoder.embedding_dim,)

    def test_batch_encode_consistency(self, encoder: YEncoder) -> None:
        """Batch encoding produces same results as single encoding."""
        texts = ["Sentence one", "Sentence two"]

        batch_embeddings = encoder.encode_batch(texts)
        single_embeddings = [encoder.encode(t) for t in texts]

        for batch_emb, single_emb in zip(batch_embeddings, single_embeddings, strict=True):
            assert np.allclose(batch_emb, single_emb, atol=1e-5)

    def test_batch_encode_empty_list(self, encoder: YEncoder) -> None:
        """Batch encoding empty list returns empty list."""
        embeddings = encoder.encode_batch([])
        assert embeddings == []

    def test_truncation_long_input(self, encoder: YEncoder) -> None:
        """Long input is truncated without error."""
        # Create input longer than max_length tokens
        long_text = "word " * 1000
        embedding = encoder.encode(long_text)
        assert embedding.shape == (encoder.embedding_dim,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_encoding_deterministic(self, encoder: YEncoder) -> None:
        """Same input produces same embedding."""
        text = "Deterministic test"
        emb1 = encoder.encode(text)
        emb2 = encoder.encode(text)
        assert np.array_equal(emb1, emb2)


class TestErrorHandling:
    """Tests for error handling."""

    def test_encode_empty_string_raises(self, encoder: YEncoder) -> None:
        """Encoding empty string raises EncodingError."""
        with pytest.raises(EncodingError, match="cannot be empty"):
            encoder.encode("")

    def test_encode_whitespace_raises(self, encoder: YEncoder) -> None:
        """Encoding whitespace-only string raises EncodingError."""
        with pytest.raises(EncodingError, match="whitespace-only"):
            encoder.encode("   \t\n  ")

    def test_batch_encode_with_empty_string_raises(self, encoder: YEncoder) -> None:
        """Batch encoding with empty string raises EncodingError."""
        with pytest.raises(EncodingError, match="index 1"):
            encoder.encode_batch(["Valid", "", "Also valid"])

    def test_invalid_model_raises(self) -> None:
        """Invalid model name raises ModelLoadError."""
        config = YEncoderConfig(
            model_name="nonexistent-model-xyz-123",
            embedding_dim=384,
        )
        with pytest.raises(ModelLoadError):
            YEncoder(config)

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched embedding_dim raises ModelLoadError."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=512,  # Wrong: MiniLM produces 384
        )
        with pytest.raises(ModelLoadError, match="dimension mismatch"):
            YEncoder(config)


class TestPoolingModes:
    """Tests for different pooling modes."""

    @pytest.mark.parametrize("pooling_mode", ["mean", "cls", "max"])
    def test_pooling_modes_produce_output(self, pooling_mode: str) -> None:
        """All pooling modes produce valid output."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            pooling_mode=pooling_mode,  # type: ignore[arg-type]
        )
        encoder = YEncoder(config)
        embedding = encoder.encode("Test pooling")
        assert embedding.shape == (384,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    @pytest.mark.parametrize("pooling_mode", ["mean", "cls", "max"])
    def test_pooling_modes_deterministic(self, pooling_mode: str) -> None:
        """All pooling modes are deterministic."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            pooling_mode=pooling_mode,  # type: ignore[arg-type]
        )
        encoder = YEncoder(config)
        text = "Determinism test"
        emb1 = encoder.encode(text)
        emb2 = encoder.encode(text)
        assert np.array_equal(emb1, emb2)


class TestSemanticQuality:
    """Tests for semantic quality of embeddings."""

    def test_similar_texts_similar_embeddings(self, encoder: YEncoder) -> None:
        """Semantically similar texts produce similar embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "A feline rested on the rug."
        text3 = "Python is a programming language."

        emb1 = encoder.encode(text1)
        emb2 = encoder.encode(text2)
        emb3 = encoder.encode(text3)

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        sim_similar = np.dot(emb1, emb2)
        sim_different = np.dot(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_similar > sim_different
        # Similar texts should have reasonably high similarity
        assert sim_similar > 0.4
        # Different texts should have lower similarity
        assert sim_different < 0.5

    def test_identical_texts_identical_embeddings(self, encoder: YEncoder) -> None:
        """Identical texts produce identical embeddings."""
        text = "Exact same text"
        emb1 = encoder.encode(text)
        emb2 = encoder.encode(text)
        assert np.array_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self, encoder: YEncoder) -> None:
        """Different texts produce different embeddings."""
        text1 = "Dogs are loyal pets"
        text2 = "Mathematical equations solve problems"
        emb1 = encoder.encode(text1)
        emb2 = encoder.encode(text2)
        assert not np.array_equal(emb1, emb2)


class TestModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_returns_dict(self, encoder: YEncoder) -> None:
        """get_model_info returns a dictionary."""
        info = encoder.get_model_info()
        assert isinstance(info, dict)

    def test_get_model_info_contains_required_keys(self, encoder: YEncoder) -> None:
        """get_model_info contains all required keys."""
        info = encoder.get_model_info()
        required_keys = {
            "model_name",
            "embedding_dim",
            "max_length",
            "pooling_mode",
            "normalize",
            "device",
            "trust_remote_code",
        }
        assert required_keys <= set(info.keys())

    def test_get_model_info_values_match_config(
        self, encoder: YEncoder, encoder_config: YEncoderConfig
    ) -> None:
        """get_model_info values match configuration."""
        info = encoder.get_model_info()
        assert info["model_name"] == encoder_config.model_name
        assert info["embedding_dim"] == encoder_config.embedding_dim
        assert info["max_length"] == encoder_config.max_length
        assert info["pooling_mode"] == encoder_config.pooling_mode
        assert info["normalize"] == encoder_config.normalize
