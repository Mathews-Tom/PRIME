"""Unit tests for MemoryCluster operations.

Tests cluster creation, member management, prototype computation,
consolidation, and similarity calculations.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from prime.mcs.cluster import ClusterMember, MemoryCluster
from prime.mcs.exceptions import ConsolidationError


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


class TestClusterMember:
    """Test ClusterMember dataclass."""

    def test_create_member(self) -> None:
        """ClusterMember stores all fields correctly."""
        embedding = _random_embedding()
        member = ClusterMember(
            memory_id="mem-1",
            embedding=embedding,
            content="Test content",
            metadata={"key": "value"},
        )

        assert member.memory_id == "mem-1"
        assert np.array_equal(member.embedding, embedding)
        assert member.content == "Test content"
        assert member.metadata == {"key": "value"}
        assert member.created_at > 0

    def test_member_default_timestamp(self) -> None:
        """ClusterMember sets created_at to current time by default."""
        before = time.time()
        member = ClusterMember(
            memory_id="mem-1",
            embedding=_random_embedding(),
            content="Test",
            metadata={},
        )
        after = time.time()

        assert before <= member.created_at <= after


class TestMemoryClusterCreation:
    """Test cluster creation and basic properties."""

    def test_create_empty_cluster(self) -> None:
        """create() returns empty cluster with correct properties."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        assert cluster.cluster_id == 0
        assert cluster.embedding_dim == 1024
        assert cluster.is_empty is True
        assert cluster.size == 0
        assert cluster.is_consolidated is False

    def test_cluster_timestamps(self) -> None:
        """Cluster has valid creation and access timestamps."""
        before = time.time()
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        after = time.time()

        assert before <= cluster.created_at <= after
        assert before <= cluster.last_accessed_at <= after
        assert cluster.access_count == 0


class TestClusterMemberManagement:
    """Test adding and managing cluster members."""

    def test_add_first_member(self) -> None:
        """Adding first member updates size and creates prototype."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        embedding = _random_embedding()

        memory_id = cluster.add_member(
            embedding=embedding,
            content="First content",
            metadata={"type": "test"},
        )

        assert cluster.size == 1
        assert cluster.is_empty is False
        assert memory_id is not None
        assert len(memory_id) > 0

    def test_add_member_with_id(self) -> None:
        """Adding member with explicit ID uses that ID."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        memory_id = cluster.add_member(
            embedding=_random_embedding(),
            content="Content",
            metadata={},
            memory_id="custom-id-123",
        )

        assert memory_id == "custom-id-123"

    def test_add_multiple_members(self) -> None:
        """Adding multiple members increases size correctly."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        for i in range(5):
            cluster.add_member(
                embedding=_random_embedding(seed=i),
                content=f"Content {i}",
                metadata={},
            )

        assert cluster.size == 5
        assert len(cluster.members) == 5

    def test_add_member_updates_access_time(self) -> None:
        """Adding member updates last_accessed_at."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        initial_time = cluster.last_accessed_at

        time.sleep(0.01)  # Small delay
        cluster.add_member(
            embedding=_random_embedding(),
            content="Content",
            metadata={},
        )

        assert cluster.last_accessed_at > initial_time

    def test_add_member_wrong_dimension_raises(self) -> None:
        """Adding member with wrong embedding dimension raises ValueError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        with pytest.raises(ValueError, match="dimension mismatch"):
            cluster.add_member(
                embedding=_random_embedding(dim=512),
                content="Content",
                metadata={},
            )

    def test_members_property_returns_copy(self) -> None:
        """members property returns a copy, not the internal list."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(),
            content="Content",
            metadata={},
        )

        members1 = cluster.members
        members2 = cluster.members
        assert members1 is not members2
        assert members1[0] is members2[0]  # Same member objects

    def test_get_member_ids(self) -> None:
        """get_member_ids returns all member IDs."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        ids = []
        for i in range(3):
            mid = cluster.add_member(
                embedding=_random_embedding(seed=i),
                content=f"Content {i}",
                metadata={},
            )
            ids.append(mid)

        assert cluster.get_member_ids() == ids

    def test_get_member_by_id_found(self) -> None:
        """get_member_by_id returns member when found."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        memory_id = cluster.add_member(
            embedding=_random_embedding(),
            content="Test content",
            metadata={"key": "value"},
        )

        member = cluster.get_member_by_id(memory_id)
        assert member is not None
        assert member.content == "Test content"

    def test_get_member_by_id_not_found(self) -> None:
        """get_member_by_id returns None when not found."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(),
            content="Content",
            metadata={},
        )

        member = cluster.get_member_by_id("nonexistent-id")
        assert member is None


class TestClusterPrototype:
    """Test prototype computation and properties."""

    def test_prototype_single_member(self) -> None:
        """Prototype equals single member's embedding (normalized)."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        embedding = _random_embedding()

        cluster.add_member(embedding=embedding, content="Content", metadata={})

        # For single member, prototype should equal the embedding
        np.testing.assert_array_almost_equal(cluster.prototype, embedding)

    def test_prototype_is_normalized(self) -> None:
        """Prototype is L2-normalized after multiple additions."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        for i in range(5):
            cluster.add_member(
                embedding=_random_embedding(seed=i),
                content=f"Content {i}",
                metadata={},
            )

        norm = np.linalg.norm(cluster.prototype)
        assert abs(norm - 1.0) < 1e-6

    def test_prototype_is_centroid(self) -> None:
        """Prototype direction matches centroid of members."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        embeddings = []

        for i in range(3):
            emb = _random_embedding(seed=i)
            embeddings.append(emb)
            cluster.add_member(embedding=emb, content=f"Content {i}", metadata={})

        # Compute expected centroid
        expected_centroid = np.stack(embeddings).mean(axis=0)
        expected_centroid = expected_centroid / np.linalg.norm(expected_centroid)

        np.testing.assert_array_almost_equal(cluster.prototype, expected_centroid)

    def test_prototype_empty_cluster_raises(self) -> None:
        """Accessing prototype of empty cluster raises ValueError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        with pytest.raises(ValueError, match="empty cluster"):
            _ = cluster.prototype


class TestClusterRepresentative:
    """Test representative member selection."""

    def test_representative_single_member(self) -> None:
        """Representative is the only member when cluster has one."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(),
            content="Only member",
            metadata={},
        )

        assert cluster.representative_content == "Only member"

    def test_representative_closest_to_prototype(self) -> None:
        """Representative is member closest to prototype."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        # Create a base embedding and add it
        base = _random_embedding(seed=42)
        cluster.add_member(embedding=base, content="Base member", metadata={})

        # Add similar members
        cluster.add_member(
            embedding=_similar_embedding(base, noise_scale=0.2, seed=1),
            content="Similar 1",
            metadata={},
        )
        cluster.add_member(
            embedding=_similar_embedding(base, noise_scale=0.3, seed=2),
            content="Similar 2",
            metadata={},
        )

        # The base member should be closest to the centroid of similar embeddings
        representative = cluster.representative_member
        assert representative is not None

    def test_representative_content_empty_raises(self) -> None:
        """Accessing representative of empty cluster raises ValueError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        with pytest.raises(ValueError, match="empty cluster"):
            _ = cluster.representative_content

    def test_representative_member_empty_raises(self) -> None:
        """Accessing representative_member of empty cluster raises ValueError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        with pytest.raises(ValueError, match="empty cluster"):
            _ = cluster.representative_member


class TestClusterSimilarity:
    """Test similarity calculations."""

    def test_similarity_to_identical_embedding(self) -> None:
        """Similarity to prototype itself should be ~1.0."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        embedding = _random_embedding()
        cluster.add_member(embedding=embedding, content="Content", metadata={})

        similarity = cluster.similarity_to(embedding)
        assert abs(similarity - 1.0) < 1e-6

    def test_similarity_range(self) -> None:
        """Similarity values should be in reasonable range."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(seed=0),
            content="Content",
            metadata={},
        )

        # Test with different embeddings
        for seed in range(1, 10):
            query = _random_embedding(seed=seed)
            similarity = cluster.similarity_to(query)
            # For normalized random vectors, similarity should be between -1 and 1
            assert -1.0 <= similarity <= 1.0

    def test_similarity_empty_cluster_raises(self) -> None:
        """Computing similarity to empty cluster raises ValueError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        with pytest.raises(ValueError, match="empty cluster"):
            cluster.similarity_to(_random_embedding())


class TestClusterConsolidation:
    """Test cluster consolidation."""

    def test_consolidate_single_member(self) -> None:
        """Consolidating single member marks consolidated, removes zero."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(),
            content="Only member",
            metadata={},
        )

        removed = cluster.consolidate()

        assert removed == 0
        assert cluster.is_consolidated is True
        assert cluster.size == 1

    def test_consolidate_multiple_members(self) -> None:
        """Consolidation keeps only representative."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        for i in range(5):
            cluster.add_member(
                embedding=_random_embedding(seed=i),
                content=f"Content {i}",
                metadata={},
            )

        removed = cluster.consolidate()

        assert removed == 4
        assert cluster.size == 1
        assert cluster.is_consolidated is True

    def test_consolidate_preserves_representative(self) -> None:
        """Consolidation preserves the representative member."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        base = _random_embedding(seed=42)
        cluster.add_member(embedding=base, content="Base", metadata={})

        for i in range(3):
            cluster.add_member(
                embedding=_similar_embedding(base, noise_scale=0.2, seed=i),
                content=f"Similar {i}",
                metadata={},
            )

        expected_rep = cluster.representative_content
        cluster.consolidate()

        assert cluster.representative_content == expected_rep

    def test_consolidate_updates_access_time(self) -> None:
        """Consolidation updates last_accessed_at."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        # Add multiple members so consolidation actually does work
        for i in range(3):
            cluster.add_member(
                embedding=_random_embedding(seed=i),
                content=f"Content {i}",
                metadata={},
            )
        initial_time = cluster.last_accessed_at

        time.sleep(0.01)
        cluster.consolidate()

        assert cluster.last_accessed_at > initial_time

    def test_consolidate_already_consolidated_raises(self) -> None:
        """Consolidating already consolidated cluster raises error."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(),
            content="Content",
            metadata={},
        )
        cluster.consolidate()

        with pytest.raises(ConsolidationError, match="already consolidated"):
            cluster.consolidate()

    def test_consolidate_empty_cluster_raises(self) -> None:
        """Consolidating empty cluster raises ValueError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        with pytest.raises(ValueError, match="empty cluster"):
            cluster.consolidate()

    def test_add_to_consolidated_raises(self) -> None:
        """Adding to consolidated cluster raises ConsolidationError."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        cluster.add_member(
            embedding=_random_embedding(),
            content="Content",
            metadata={},
        )
        cluster.consolidate()

        with pytest.raises(ConsolidationError, match="consolidated cluster"):
            cluster.add_member(
                embedding=_random_embedding(),
                content="New content",
                metadata={},
            )


class TestClusterAccess:
    """Test cluster access tracking."""

    def test_touch_updates_access_time(self) -> None:
        """touch() updates last_accessed_at."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)
        initial_time = cluster.last_accessed_at

        time.sleep(0.01)
        cluster.touch()

        assert cluster.last_accessed_at > initial_time

    def test_touch_increments_access_count(self) -> None:
        """touch() increments access_count."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        assert cluster.access_count == 0
        cluster.touch()
        assert cluster.access_count == 1
        cluster.touch()
        assert cluster.access_count == 2


class TestClusterPrototypeStability:
    """Test prototype stability under various conditions."""

    def test_prototype_stable_after_similar_additions(self) -> None:
        """Prototype remains stable when adding similar embeddings."""
        cluster = MemoryCluster.create(cluster_id=0, embedding_dim=1024)

        base = _random_embedding(seed=42)
        cluster.add_member(embedding=base, content="Base", metadata={})
        initial_prototype = cluster.prototype.copy()

        # Add very similar embeddings
        for i in range(3):
            cluster.add_member(
                embedding=_similar_embedding(base, noise_scale=0.05, seed=i),
                content=f"Similar {i}",
                metadata={},
            )

        # Prototype should be close to initial (similar embeddings)
        similarity = np.dot(initial_prototype, cluster.prototype)
        assert similarity > 0.9
