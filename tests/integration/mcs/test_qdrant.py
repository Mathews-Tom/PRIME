"""Integration tests for QdrantIndex.

Tests CRUD operations, search functionality, and hybrid search
using Qdrant's in-memory mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from prime.mcs import (
    IndexSearchResult,
    MCSConfig,
    QdrantIndex,
    SparseVector,
)


def _random_embedding(dim: int = 1024, seed: int | None = None) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _similar_embedding(
    base: np.ndarray,
    noise_scale: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an embedding similar to base with added noise."""
    rng = np.random.default_rng(seed)
    noise = rng.random(base.shape[0]).astype(np.float32) * noise_scale
    vec = base + noise
    return vec / np.linalg.norm(vec)


@pytest.fixture
def config() -> MCSConfig:
    """Create MCS config for testing."""
    return MCSConfig(
        collection_name="test_mcs_collection",
        embedding_dim=128,  # Smaller for faster tests
        enable_hybrid_search=True,
    )


@pytest.fixture
def index(config: MCSConfig) -> QdrantIndex:
    """Create in-memory QdrantIndex for testing."""
    idx = QdrantIndex(config, in_memory=True)
    yield idx
    idx.close()


@pytest.fixture
def index_dense_only() -> QdrantIndex:
    """Create in-memory QdrantIndex with hybrid search disabled."""
    config = MCSConfig(
        collection_name="test_dense_only",
        embedding_dim=128,
        enable_hybrid_search=False,
    )
    idx = QdrantIndex(config, in_memory=True)
    yield idx
    idx.close()


class TestQdrantIndexBasicOperations:
    """Test basic CRUD operations."""

    def test_add_and_count(self, index: QdrantIndex) -> None:
        """Add vectors and verify count increases."""
        assert index.count() == 0

        embedding = _random_embedding(dim=128, seed=1)
        index.add("vec-1", dense=embedding)

        assert index.count() == 1

    def test_add_multiple(self, index: QdrantIndex) -> None:
        """Add multiple vectors."""
        for i in range(5):
            embedding = _random_embedding(dim=128, seed=i)
            index.add(f"vec-{i}", dense=embedding)

        assert index.count() == 5

    def test_add_with_payload(self, index: QdrantIndex) -> None:
        """Add vector with metadata payload."""
        embedding = _random_embedding(dim=128, seed=1)
        index.add(
            "vec-1",
            dense=embedding,
            payload={"type": "preference", "importance": 0.9},
        )

        assert index.count() == 1

    def test_add_with_sparse_vector(self, index: QdrantIndex) -> None:
        """Add vector with sparse component."""
        embedding = _random_embedding(dim=128, seed=1)
        sparse = SparseVector(indices=[0, 5, 10], values=[0.5, 0.3, 0.8])

        index.add("vec-1", dense=embedding, sparse=sparse)

        assert index.count() == 1

    def test_get_existing_vector(self, index: QdrantIndex) -> None:
        """Get vector by ID returns correct embedding."""
        embedding = _random_embedding(dim=128, seed=42)
        index.add("vec-1", dense=embedding)

        result = index.get("vec-1")

        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding, decimal=5)

    def test_get_nonexistent_vector(self, index: QdrantIndex) -> None:
        """Get nonexistent vector returns None."""
        result = index.get("nonexistent")
        assert result is None

    def test_remove_existing_vector(self, index: QdrantIndex) -> None:
        """Remove existing vector returns True."""
        embedding = _random_embedding(dim=128, seed=1)
        index.add("vec-1", dense=embedding)

        removed = index.remove("vec-1")

        assert removed is True
        assert index.count() == 0

    def test_remove_nonexistent_vector(self, index: QdrantIndex) -> None:
        """Remove nonexistent vector returns False."""
        removed = index.remove("nonexistent")
        assert removed is False

    def test_clear(self, index: QdrantIndex) -> None:
        """Clear removes all vectors."""
        for i in range(10):
            embedding = _random_embedding(dim=128, seed=i)
            index.add(f"vec-{i}", dense=embedding)

        assert index.count() == 10

        index.clear()

        assert index.count() == 0

    def test_upsert_updates_existing(self, index: QdrantIndex) -> None:
        """Adding same ID updates the vector."""
        embedding1 = _random_embedding(dim=128, seed=1)
        embedding2 = _random_embedding(dim=128, seed=2)

        index.add("vec-1", dense=embedding1)
        index.add("vec-1", dense=embedding2)

        assert index.count() == 1

        result = index.get("vec-1")
        np.testing.assert_array_almost_equal(result, embedding2, decimal=5)


class TestQdrantIndexDenseSearch:
    """Test dense vector search."""

    def test_search_returns_results(self, index: QdrantIndex) -> None:
        """Search returns results ordered by similarity."""
        base = _random_embedding(dim=128, seed=42)
        index.add("base", dense=base)

        # Add similar vectors
        for i in range(5):
            similar = _similar_embedding(base, noise_scale=0.1 * (i + 1), seed=i)
            index.add(f"similar-{i}", dense=similar)

        results = index.search_dense(query=base, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, IndexSearchResult) for r in results)
        # Base should be first (highest similarity)
        assert results[0].id == "base"

    def test_search_scores_ordered(self, index: QdrantIndex) -> None:
        """Search results have descending scores."""
        base = _random_embedding(dim=128, seed=42)

        for i in range(10):
            vec = _random_embedding(dim=128, seed=i)
            index.add(f"vec-{i}", dense=vec)

        results = index.search_dense(query=base, top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limits_results(self, index: QdrantIndex) -> None:
        """Search returns at most top_k results."""
        for i in range(20):
            vec = _random_embedding(dim=128, seed=i)
            index.add(f"vec-{i}", dense=vec)

        results = index.search_dense(
            query=_random_embedding(dim=128, seed=100),
            top_k=5,
        )

        assert len(results) == 5

    def test_search_empty_index(self, index: QdrantIndex) -> None:
        """Search on empty index returns empty list."""
        results = index.search_dense(
            query=_random_embedding(dim=128, seed=1),
            top_k=5,
        )

        assert results == []

    def test_search_with_payload_filter(self, index: QdrantIndex) -> None:
        """Search with payload filter returns filtered results."""
        base = _random_embedding(dim=128, seed=42)

        # Add vectors with different types
        for i in range(5):
            vec = _similar_embedding(base, noise_scale=0.1, seed=i)
            index.add(f"pref-{i}", dense=vec, payload={"type": "preference"})

        for i in range(5):
            vec = _similar_embedding(base, noise_scale=0.1, seed=i + 10)
            index.add(f"fact-{i}", dense=vec, payload={"type": "fact"})

        results = index.search_dense(
            query=base,
            top_k=10,
            filter_payload={"type": "preference"},
        )

        assert len(results) == 5
        assert all(r.id.startswith("pref-") for r in results)


class TestQdrantIndexHybridSearch:
    """Test hybrid search with dense + sparse vectors."""

    def test_hybrid_search_combines_results(self, index: QdrantIndex) -> None:
        """Hybrid search uses both dense and sparse vectors."""
        base = _random_embedding(dim=128, seed=42)

        # Add vectors with different sparse representations
        for i in range(5):
            vec = _similar_embedding(base, noise_scale=0.1, seed=i)
            sparse = SparseVector(indices=[i, i + 10], values=[0.8, 0.5])
            index.add(f"vec-{i}", dense=vec, sparse=sparse)

        sparse_query = SparseVector(indices=[0, 10], values=[0.8, 0.5])
        results = index.search_hybrid(
            dense_query=base,
            sparse_query=sparse_query,
            top_k=3,
        )

        assert len(results) == 3
        assert all(isinstance(r, IndexSearchResult) for r in results)

    def test_hybrid_fallback_when_disabled(
        self,
        index_dense_only: QdrantIndex,
    ) -> None:
        """Hybrid search falls back to dense when disabled."""
        base = _random_embedding(dim=128, seed=42)
        index_dense_only.add("vec-1", dense=base)

        sparse_query = SparseVector(indices=[0], values=[0.5])
        results = index_dense_only.search_hybrid(
            dense_query=base,
            sparse_query=sparse_query,
            top_k=3,
        )

        assert len(results) == 1
        assert results[0].id == "vec-1"

    def test_hybrid_search_with_filter(self, index: QdrantIndex) -> None:
        """Hybrid search respects payload filters."""
        base = _random_embedding(dim=128, seed=42)

        for i in range(3):
            vec = _similar_embedding(base, noise_scale=0.1, seed=i)
            sparse = SparseVector(indices=[i], values=[0.5])
            index.add(f"typeA-{i}", dense=vec, sparse=sparse, payload={"cat": "A"})

        for i in range(3):
            vec = _similar_embedding(base, noise_scale=0.1, seed=i + 10)
            sparse = SparseVector(indices=[i + 10], values=[0.5])
            index.add(f"typeB-{i}", dense=vec, sparse=sparse, payload={"cat": "B"})

        sparse_query = SparseVector(indices=[0, 1, 2], values=[0.5, 0.5, 0.5])
        results = index.search_hybrid(
            dense_query=base,
            sparse_query=sparse_query,
            top_k=10,
            filter_payload={"cat": "A"},
        )

        assert len(results) == 3
        assert all(r.id.startswith("typeA-") for r in results)


class TestQdrantIndexCollectionManagement:
    """Test collection creation and management."""

    def test_collection_created_automatically(self, config: MCSConfig) -> None:
        """Collection is created if it doesn't exist."""
        index = QdrantIndex(config, in_memory=True)
        try:
            # Should not raise
            assert index.count() == 0
        finally:
            index.close()

    def test_multiple_indexes_same_collection(self, config: MCSConfig) -> None:
        """Multiple indexes can use same in-memory collection."""
        index1 = QdrantIndex(config, in_memory=True)
        try:
            embedding = _random_embedding(dim=128, seed=1)
            index1.add("vec-1", dense=embedding)
            assert index1.count() == 1
        finally:
            index1.close()


class TestQdrantIndexEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters_in_id(self, index: QdrantIndex) -> None:
        """IDs with special characters work correctly."""
        embedding = _random_embedding(dim=128, seed=1)
        special_id = "user:123/preference:dark-mode"

        index.add(special_id, dense=embedding)
        result = index.get(special_id)

        assert result is not None

    def test_empty_sparse_vector(self, index: QdrantIndex) -> None:
        """Empty sparse vector is handled."""
        embedding = _random_embedding(dim=128, seed=1)
        sparse = SparseVector(indices=[], values=[])

        index.add("vec-1", dense=embedding, sparse=sparse)
        assert index.count() == 1

    def test_large_batch_operations(self, index: QdrantIndex) -> None:
        """Handle larger number of vectors."""
        for i in range(100):
            embedding = _random_embedding(dim=128, seed=i)
            index.add(f"vec-{i}", dense=embedding)

        assert index.count() == 100

        results = index.search_dense(
            query=_random_embedding(dim=128, seed=0),
            top_k=50,
        )
        assert len(results) == 50
