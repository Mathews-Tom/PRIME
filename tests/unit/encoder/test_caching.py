"""Unit tests for YEncoder caching functionality."""

from __future__ import annotations

import numpy as np
import pytest

from prime.encoder import CacheInfo, YEncoder, YEncoderConfig


class TestCacheDisabled:
    """Tests for encoder with caching disabled."""

    @pytest.fixture
    def encoder_no_cache(self) -> YEncoder:
        """Encoder with cache disabled."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            cache_size=0,
        )
        return YEncoder(config)

    def test_cache_disabled_by_default(self, encoder_no_cache: YEncoder) -> None:
        """Cache is disabled when cache_size=0."""
        info = encoder_no_cache.get_model_info()
        assert info["cache_enabled"] is False
        assert "cache_info" not in info

    def test_cache_info_returns_none_when_disabled(
        self, encoder_no_cache: YEncoder
    ) -> None:
        """cache_info() returns None when caching disabled."""
        assert encoder_no_cache.cache_info() is None

    def test_clear_cache_safe_when_disabled(
        self, encoder_no_cache: YEncoder
    ) -> None:
        """clear_cache() is safe to call when caching disabled."""
        encoder_no_cache.clear_cache()  # Should not raise


class TestCacheEnabled:
    """Tests for encoder with caching enabled."""

    @pytest.fixture
    def encoder_with_cache(self) -> YEncoder:
        """Encoder with cache enabled."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            cache_size=100,
        )
        return YEncoder(config)

    def test_cache_enabled_in_model_info(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """Cache enabled flag is in model info."""
        info = encoder_with_cache.get_model_info()
        assert info["cache_enabled"] is True

    def test_cache_info_returns_cache_info(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """cache_info() returns CacheInfo when enabled."""
        info = encoder_with_cache.cache_info()
        assert info is not None
        assert isinstance(info, CacheInfo)
        assert info.maxsize == 100
        assert info.currsize == 0
        assert info.hits == 0
        assert info.misses == 0

    def test_first_encode_is_cache_miss(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """First encoding is a cache miss."""
        encoder_with_cache.encode("Test text")
        info = encoder_with_cache.cache_info()
        assert info is not None
        assert info.misses == 1
        assert info.hits == 0

    def test_second_encode_same_text_is_cache_hit(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """Second encoding of same text is cache hit."""
        text = "Cached text"
        encoder_with_cache.encode(text)
        encoder_with_cache.encode(text)
        info = encoder_with_cache.cache_info()
        assert info is not None
        assert info.hits == 1
        assert info.misses == 1

    def test_cached_result_matches_original(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """Cached result matches original encoding."""
        text = "Compare cached vs uncached"
        emb1 = encoder_with_cache.encode(text)
        emb2 = encoder_with_cache.encode(text)
        assert np.array_equal(emb1, emb2)

    def test_different_texts_are_cached_separately(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """Different texts have separate cache entries."""
        encoder_with_cache.encode("Text A")
        encoder_with_cache.encode("Text B")
        info = encoder_with_cache.cache_info()
        assert info is not None
        assert info.currsize == 2
        assert info.misses == 2

    def test_clear_cache_resets_stats(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """clear_cache() resets cache to empty state."""
        encoder_with_cache.encode("Text to cache")
        encoder_with_cache.encode("Text to cache")  # Hit
        encoder_with_cache.clear_cache()
        info = encoder_with_cache.cache_info()
        assert info is not None
        assert info.currsize == 0
        assert info.hits == 0
        assert info.misses == 0

    def test_cache_eviction_at_maxsize(self) -> None:
        """Cache evicts oldest entries when full."""
        config = YEncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            cache_size=3,  # Small cache for testing
        )
        encoder = YEncoder(config)

        # Fill cache
        encoder.encode("Text 1")
        encoder.encode("Text 2")
        encoder.encode("Text 3")

        info = encoder.cache_info()
        assert info is not None
        assert info.currsize == 3

        # Add one more, should evict oldest
        encoder.encode("Text 4")
        info = encoder.cache_info()
        assert info is not None
        # LRU cache maintains maxsize
        assert info.currsize == 3

    def test_cache_key_uses_sha256(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """Cache key computation uses SHA256."""
        import hashlib

        text = "Test text"
        expected_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        actual_key = encoder_with_cache._compute_cache_key(text)
        assert actual_key == expected_key

    def test_normalized_output_still_normalized_from_cache(
        self, encoder_with_cache: YEncoder
    ) -> None:
        """Cached embeddings maintain L2 normalization."""
        text = "Normalization test"
        emb1 = encoder_with_cache.encode(text)  # Miss
        emb2 = encoder_with_cache.encode(text)  # Hit

        assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(emb2), 1.0, atol=1e-5)
