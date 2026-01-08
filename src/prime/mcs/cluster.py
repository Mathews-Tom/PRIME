"""Memory cluster implementation for MCS.

Provides the MemoryCluster class for managing semantically similar
memories with automatic prototype computation and consolidation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from prime.mcs.exceptions import ConsolidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ClusterMember:
    """Individual member within a memory cluster.

    Stores the embedding, content, and metadata for a single memory
    that has been assigned to a cluster.

    Attributes:
        memory_id: Unique identifier for this memory.
        embedding: L2-normalized embedding vector.
        content: Original text content.
        metadata: Key-value metadata dictionary.
        created_at: Unix timestamp of creation.
    """

    memory_id: str
    embedding: NDArray[np.float32]
    content: str
    metadata: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryCluster:
    """A cluster of semantically similar memories.

    Manages a collection of memories that share semantic similarity.
    Maintains an L2-normalized prototype (centroid) that is updated
    as members are added. Supports consolidation to compress storage.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        created_at: Unix timestamp of cluster creation.
        last_accessed_at: Unix timestamp of last access.
        access_count: Number of times cluster was accessed.

    Example:
        >>> cluster = MemoryCluster.create(embedding_dim=1024)
        >>> cluster.add_member(embedding, "User likes dark mode", {})
        >>> cluster.add_member(embedding2, "Prefers night theme", {})
        >>> print(f"Cluster size: {cluster.size}")
        >>> print(f"Representative: {cluster.representative_content}")
    """

    cluster_id: int
    embedding_dim: int
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    _members: list[ClusterMember] = field(default_factory=list)
    _prototype: NDArray[np.float32] | None = field(default=None, repr=False)
    _consolidated: bool = field(default=False)

    @classmethod
    def create(cls, cluster_id: int, embedding_dim: int) -> MemoryCluster:
        """Create a new empty cluster.

        Args:
            cluster_id: Unique identifier for the cluster.
            embedding_dim: Dimension of embedding vectors.

        Returns:
            New MemoryCluster instance.
        """
        return cls(cluster_id=cluster_id, embedding_dim=embedding_dim)

    @property
    def size(self) -> int:
        """Return number of members in cluster."""
        return len(self._members)

    @property
    def is_empty(self) -> bool:
        """Return True if cluster has no members."""
        return len(self._members) == 0

    @property
    def is_consolidated(self) -> bool:
        """Return True if cluster has been consolidated."""
        return self._consolidated

    @property
    def prototype(self) -> NDArray[np.float32]:
        """Return L2-normalized cluster prototype.

        Raises:
            ValueError: If cluster is empty.
        """
        if self._prototype is None:
            msg = "Cannot get prototype of empty cluster"
            raise ValueError(msg)
        return self._prototype

    @property
    def members(self) -> list[ClusterMember]:
        """Return list of cluster members (read-only copy)."""
        return list(self._members)

    @property
    def representative_content(self) -> str:
        """Return content of member closest to prototype.

        The representative is the member whose embedding has the
        highest cosine similarity to the cluster prototype.

        Returns:
            Content string of the representative member.

        Raises:
            ValueError: If cluster is empty.
        """
        if self.is_empty:
            msg = "Cannot get representative of empty cluster"
            raise ValueError(msg)

        if len(self._members) == 1:
            return self._members[0].content

        # Find member with highest similarity to prototype
        similarities = [
            self._cosine_similarity(member.embedding, self._prototype)
            for member in self._members
        ]
        representative_idx = int(np.argmax(similarities))
        return self._members[representative_idx].content

    @property
    def representative_member(self) -> ClusterMember:
        """Return member closest to prototype.

        Returns:
            ClusterMember instance of the representative.

        Raises:
            ValueError: If cluster is empty.
        """
        if self.is_empty:
            msg = "Cannot get representative of empty cluster"
            raise ValueError(msg)

        if len(self._members) == 1:
            return self._members[0]

        similarities = [
            self._cosine_similarity(member.embedding, self._prototype)
            for member in self._members
        ]
        representative_idx = int(np.argmax(similarities))
        return self._members[representative_idx]

    def add_member(
        self,
        embedding: NDArray[np.float32],
        content: str,
        metadata: dict[str, Any],
        memory_id: str | None = None,
    ) -> str:
        """Add a member to the cluster and update prototype.

        The prototype is recomputed as the L2-normalized centroid
        of all member embeddings after addition.

        Args:
            embedding: L2-normalized embedding vector.
            content: Text content to store.
            metadata: Key-value metadata dictionary.
            memory_id: Optional unique ID. Generated if not provided.

        Returns:
            The memory_id of the added member.

        Raises:
            ConsolidationError: If cluster has been consolidated.
            ValueError: If embedding dimension doesn't match.
        """
        if self._consolidated:
            raise ConsolidationError(
                self.cluster_id,
                "Cannot add member to consolidated cluster",
            )

        if embedding.shape[0] != self.embedding_dim:
            msg = (
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )
            raise ValueError(msg)

        if memory_id is None:
            memory_id = str(uuid.uuid4())

        member = ClusterMember(
            memory_id=memory_id,
            embedding=embedding,
            content=content,
            metadata=metadata,
        )
        self._members.append(member)
        self._update_prototype()
        self.last_accessed_at = time.time()

        return memory_id

    def similarity_to(self, embedding: NDArray[np.float32]) -> float:
        """Compute cosine similarity between embedding and prototype.

        Args:
            embedding: Query embedding vector.

        Returns:
            Cosine similarity in range [0, 1] for normalized vectors.

        Raises:
            ValueError: If cluster is empty.
        """
        if self.is_empty:
            msg = "Cannot compute similarity to empty cluster"
            raise ValueError(msg)

        return float(self._cosine_similarity(embedding, self._prototype))

    def consolidate(self) -> int:
        """Consolidate cluster by keeping only the representative.

        After consolidation, the cluster is marked immutable and only
        contains the single member closest to the prototype. This
        achieves storage compression while preserving retrieval quality.

        Returns:
            Number of members removed during consolidation.

        Raises:
            ConsolidationError: If cluster is already consolidated.
            ValueError: If cluster is empty.
        """
        if self._consolidated:
            raise ConsolidationError(
                self.cluster_id,
                "Cluster is already consolidated",
            )

        if self.is_empty:
            msg = "Cannot consolidate empty cluster"
            raise ValueError(msg)

        if len(self._members) == 1:
            # Nothing to consolidate, just mark as consolidated
            self._consolidated = True
            return 0

        # Find and keep only representative
        representative = self.representative_member
        members_removed = len(self._members) - 1
        self._members = [representative]
        self._consolidated = True
        self.last_accessed_at = time.time()

        return members_removed

    def touch(self) -> None:
        """Update last_accessed_at and increment access_count."""
        self.last_accessed_at = time.time()
        self.access_count += 1

    def get_member_ids(self) -> list[str]:
        """Return list of all member memory IDs."""
        return [member.memory_id for member in self._members]

    def get_member_by_id(self, memory_id: str) -> ClusterMember | None:
        """Find member by memory_id.

        Args:
            memory_id: The memory ID to search for.

        Returns:
            ClusterMember if found, None otherwise.
        """
        for member in self._members:
            if member.memory_id == memory_id:
                return member
        return None

    def _update_prototype(self) -> None:
        """Recompute prototype as L2-normalized centroid of all members."""
        if self.is_empty:
            self._prototype = None
            return

        embeddings = np.stack([m.embedding for m in self._members])
        centroid = embeddings.mean(axis=0)

        # L2 normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            self._prototype = (centroid / norm).astype(np.float32)
        else:
            # Degenerate case: zero centroid (should not happen with valid embeddings)
            self._prototype = centroid.astype(np.float32)

    @staticmethod
    def _cosine_similarity(
        a: NDArray[np.float32],
        b: NDArray[np.float32] | None,
    ) -> float:
        """Compute cosine similarity between two vectors.

        For L2-normalized vectors, this is equivalent to dot product.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity value.
        """
        if b is None:
            return 0.0
        return float(np.dot(a, b))
