"""Unit tests for MCS configuration validation.

Tests configuration schema validation, default values, and
preset configurations for Memory Cluster Store.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prime.mcs import (
    COMPACT_CONFIG,
    DEFAULT_CONFIG,
    HIGH_PRECISION_CONFIG,
    MEMORY_EFFICIENT_CONFIG,
    MCSConfig,
)


class TestMCSConfigDefaults:
    """Test default configuration values."""

    def test_default_embedding_dim(self) -> None:
        """Default embedding dimension should be 1024."""
        config = MCSConfig()
        assert config.embedding_dim == 1024

    def test_default_similarity_threshold(self) -> None:
        """Default similarity threshold should be 0.85."""
        config = MCSConfig()
        assert config.similarity_threshold == 0.85

    def test_default_consolidation_threshold(self) -> None:
        """Default consolidation threshold should be 5."""
        config = MCSConfig()
        assert config.consolidation_threshold == 5

    def test_default_max_clusters(self) -> None:
        """Default max clusters should be 10000."""
        config = MCSConfig()
        assert config.max_clusters == 10000

    def test_default_decay_factor(self) -> None:
        """Default decay factor should be 0.99."""
        config = MCSConfig()
        assert config.decay_factor == 0.99

    def test_default_decay_unit_seconds(self) -> None:
        """Default decay unit should be 3600 seconds (1 hour)."""
        config = MCSConfig()
        assert config.decay_unit_seconds == 3600

    def test_default_index_type(self) -> None:
        """Default index type should be qdrant."""
        config = MCSConfig()
        assert config.index_type == "qdrant"

    def test_default_hybrid_search_enabled(self) -> None:
        """Hybrid search should be enabled by default."""
        config = MCSConfig()
        assert config.enable_hybrid_search is True

    def test_default_sparse_weight(self) -> None:
        """Default sparse weight should be 0.3."""
        config = MCSConfig()
        assert config.sparse_weight == 0.3

    def test_default_fusion_k(self) -> None:
        """Default RRF fusion k should be 60."""
        config = MCSConfig()
        assert config.fusion_k == 60

    def test_default_qdrant_settings(self) -> None:
        """Default Qdrant settings should be localhost:6333."""
        config = MCSConfig()
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.qdrant_api_key is None
        assert config.qdrant_prefer_grpc is True


class TestMCSConfigValidation:
    """Test configuration validation constraints."""

    def test_embedding_dim_must_be_positive(self) -> None:
        """Embedding dimension must be greater than 0."""
        with pytest.raises(ValidationError) as exc_info:
            MCSConfig(embedding_dim=0)
        assert "greater than 0" in str(exc_info.value).lower()

    def test_embedding_dim_negative_invalid(self) -> None:
        """Negative embedding dimension should be rejected."""
        with pytest.raises(ValidationError):
            MCSConfig(embedding_dim=-1)

    def test_similarity_threshold_range(self) -> None:
        """Similarity threshold must be in [0, 1]."""
        MCSConfig(similarity_threshold=0.0)
        MCSConfig(similarity_threshold=1.0)

        with pytest.raises(ValidationError):
            MCSConfig(similarity_threshold=-0.1)
        with pytest.raises(ValidationError):
            MCSConfig(similarity_threshold=1.1)

    def test_consolidation_threshold_minimum(self) -> None:
        """Consolidation threshold must be at least 2."""
        MCSConfig(consolidation_threshold=2)

        with pytest.raises(ValidationError):
            MCSConfig(consolidation_threshold=1)

    def test_max_clusters_minimum(self) -> None:
        """Max clusters must be at least 100."""
        MCSConfig(max_clusters=100)

        with pytest.raises(ValidationError):
            MCSConfig(max_clusters=99)

    def test_decay_factor_range(self) -> None:
        """Decay factor must be in [0, 1]."""
        MCSConfig(decay_factor=0.0)
        MCSConfig(decay_factor=1.0)

        with pytest.raises(ValidationError):
            MCSConfig(decay_factor=-0.01)
        with pytest.raises(ValidationError):
            MCSConfig(decay_factor=1.01)

    def test_decay_unit_seconds_minimum(self) -> None:
        """Decay unit must be at least 1 second."""
        MCSConfig(decay_unit_seconds=1)

        with pytest.raises(ValidationError):
            MCSConfig(decay_unit_seconds=0)

    def test_index_type_literal(self) -> None:
        """Index type must be 'qdrant' or 'faiss'."""
        MCSConfig(index_type="qdrant")
        MCSConfig(index_type="faiss")

        with pytest.raises(ValidationError):
            MCSConfig(index_type="invalid")  # type: ignore[arg-type]

    def test_collection_name_not_empty(self) -> None:
        """Collection name cannot be empty."""
        with pytest.raises(ValidationError):
            MCSConfig(collection_name="")

    def test_sparse_weight_range(self) -> None:
        """Sparse weight must be in [0, 1]."""
        MCSConfig(sparse_weight=0.0)
        MCSConfig(sparse_weight=1.0)

        with pytest.raises(ValidationError):
            MCSConfig(sparse_weight=-0.1)
        with pytest.raises(ValidationError):
            MCSConfig(sparse_weight=1.1)

    def test_fusion_k_minimum(self) -> None:
        """Fusion k must be at least 1."""
        MCSConfig(fusion_k=1)

        with pytest.raises(ValidationError):
            MCSConfig(fusion_k=0)

    def test_prefetch_multiplier_range(self) -> None:
        """Prefetch multiplier must be in [1, 10]."""
        MCSConfig(prefetch_multiplier=1)
        MCSConfig(prefetch_multiplier=10)

        with pytest.raises(ValidationError):
            MCSConfig(prefetch_multiplier=0)
        with pytest.raises(ValidationError):
            MCSConfig(prefetch_multiplier=11)

    def test_qdrant_port_range(self) -> None:
        """Qdrant port must be valid port number."""
        MCSConfig(qdrant_port=1)
        MCSConfig(qdrant_port=65535)

        with pytest.raises(ValidationError):
            MCSConfig(qdrant_port=0)
        with pytest.raises(ValidationError):
            MCSConfig(qdrant_port=65536)


class TestMCSConfigImmutability:
    """Test configuration immutability."""

    def test_config_is_frozen(self) -> None:
        """Configuration should be immutable after creation."""
        config = MCSConfig()

        with pytest.raises(ValidationError):
            config.embedding_dim = 512  # type: ignore[misc]

    def test_config_hashable(self) -> None:
        """Frozen config should be hashable."""
        config = MCSConfig()
        hash(config)


class TestPresetConfigurations:
    """Test preset configuration values."""

    def test_default_config_valid(self) -> None:
        """DEFAULT_CONFIG should have valid defaults."""
        assert DEFAULT_CONFIG.embedding_dim == 1024
        assert DEFAULT_CONFIG.similarity_threshold == 0.85
        assert DEFAULT_CONFIG.enable_hybrid_search is True

    def test_compact_config(self) -> None:
        """COMPACT_CONFIG should favor smaller clusters."""
        assert COMPACT_CONFIG.similarity_threshold == 0.90
        assert COMPACT_CONFIG.consolidation_threshold == 3
        assert COMPACT_CONFIG.max_clusters == 1000
        assert COMPACT_CONFIG.decay_factor == 0.95

    def test_high_precision_config(self) -> None:
        """HIGH_PRECISION_CONFIG should favor retrieval quality."""
        assert HIGH_PRECISION_CONFIG.similarity_threshold == 0.80
        assert HIGH_PRECISION_CONFIG.consolidation_threshold == 10
        assert HIGH_PRECISION_CONFIG.max_clusters == 50000
        assert HIGH_PRECISION_CONFIG.sparse_weight == 0.4

    def test_memory_efficient_config(self) -> None:
        """MEMORY_EFFICIENT_CONFIG should minimize resource usage."""
        assert MEMORY_EFFICIENT_CONFIG.embedding_dim == 384
        assert MEMORY_EFFICIENT_CONFIG.max_clusters == 5000
        assert MEMORY_EFFICIENT_CONFIG.enable_hybrid_search is False


class TestMCSConfigCustomization:
    """Test custom configuration scenarios."""

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        config = MCSConfig(
            similarity_threshold=0.92,
            consolidation_threshold=8,
        )
        assert config.similarity_threshold == 0.92
        assert config.consolidation_threshold == 8

    def test_custom_qdrant_cloud(self) -> None:
        """Qdrant cloud configuration should work."""
        config = MCSConfig(
            qdrant_host="my-cluster.qdrant.io",
            qdrant_port=6334,
            qdrant_api_key="secret-api-key",
        )
        assert config.qdrant_host == "my-cluster.qdrant.io"
        assert config.qdrant_api_key == "secret-api-key"

    def test_disable_hybrid_search(self) -> None:
        """Hybrid search can be disabled."""
        config = MCSConfig(enable_hybrid_search=False)
        assert config.enable_hybrid_search is False

    def test_faiss_index_type(self) -> None:
        """FAISS index type should be valid."""
        config = MCSConfig(index_type="faiss")
        assert config.index_type == "faiss"

    def test_aggressive_decay(self) -> None:
        """Aggressive decay settings for short-term memory."""
        config = MCSConfig(
            decay_factor=0.90,
            decay_unit_seconds=60,
        )
        assert config.decay_factor == 0.90
        assert config.decay_unit_seconds == 60
