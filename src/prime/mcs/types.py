"""Type definitions for Memory Cluster Store.

Defines input/output models for MCS operations including write,
read, and search results with Pydantic validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search mode for memory retrieval.

    Controls how search queries are executed against the index.

    Modes:
        DENSE: Vector-only search using cosine similarity.
        SPARSE: BM25/keyword-only search using sparse vectors.
        HYBRID: Combined dense + sparse with RRF fusion.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class MemoryWriteInput(BaseModel):
    """Input for memory write operation.

    Specifies content and metadata to be stored in MCS.
    The embedding is computed internally using the Y-Encoder.

    Attributes:
        content: Text content to store (1-10000 characters).
        metadata: Optional key-value metadata for filtering.
        user_id: Optional user identifier for scoping.
        session_id: Optional session identifier for grouping.
        force_new_cluster: If True, always create a new cluster.

    Example:
        >>> write_input = MemoryWriteInput(
        ...     content="The user prefers dark mode.",
        ...     metadata={"type": "preference", "category": "ui"},
        ...     session_id="session-123",
        ... )
    """

    content: str = Field(
        min_length=1,
        max_length=10000,
        description="Text content to store",
    )
    metadata: dict[str, str | int | float | bool] | None = Field(
        default=None,
        description="Optional metadata for filtering",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user identifier for scoping",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for grouping",
    )
    force_new_cluster: bool = Field(
        default=False,
        description="Force creation of new cluster",
    )

    model_config = {"frozen": True}


class MemoryWriteResult(BaseModel):
    """Result of memory write operation.

    Contains identifiers and status of the write operation
    including cluster assignment information.

    Attributes:
        memory_id: Unique identifier for the stored memory.
        cluster_id: ID of the cluster the memory was assigned to.
        is_new_cluster: True if a new cluster was created.
        consolidated: True if consolidation was triggered.
        similarity_to_prototype: Cosine similarity to cluster prototype.

    Example:
        >>> result = mcs.write(write_input)
        >>> print(f"Stored in cluster {result.cluster_id}")
        >>> if result.consolidated:
        ...     print("Cluster was consolidated")
    """

    memory_id: str = Field(
        description="Unique memory identifier",
    )
    cluster_id: int = Field(
        ge=0,
        description="Assigned cluster ID",
    )
    is_new_cluster: bool = Field(
        description="True if new cluster was created",
    )
    consolidated: bool = Field(
        description="True if consolidation was triggered",
    )
    similarity_to_prototype: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity to cluster prototype",
    )

    model_config = {"frozen": True}


class MemoryReadInput(BaseModel):
    """Input for memory read/search operation.

    Specifies search parameters for retrieving memories.
    Supports dense, sparse, and hybrid search modes.

    Attributes:
        embedding: Query embedding vector (must match embedding_dim).
        query_text: Original query text for BM25 search.
        k: Number of results to return (1-100).
        user_id: Optional user filter for scoping.
        session_id: Optional session filter for grouping.
        min_similarity: Minimum similarity threshold for results.
        search_mode: Search mode (dense, sparse, hybrid).
        sparse_weight: Weight for sparse vectors in hybrid mode.

    Example:
        >>> read_input = MemoryReadInput(
        ...     embedding=query_embedding.tolist(),
        ...     query_text="What are the user's preferences?",
        ...     k=5,
        ...     search_mode=SearchMode.HYBRID,
        ... )
    """

    embedding: list[float] = Field(
        description="Query embedding vector",
    )
    query_text: str | None = Field(
        default=None,
        description="Original query text for BM25 search",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user filter",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session filter",
    )
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode: dense, sparse, or hybrid",
    )
    sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for sparse vectors in hybrid mode",
    )

    model_config = {"frozen": True}


class MemoryReadResult(BaseModel):
    """Single memory retrieval result.

    Contains the retrieved memory content, metadata, and
    relevance scores including decay-adjusted ranking.

    Attributes:
        memory_id: Unique identifier of the retrieved memory.
        content: Original text content of the memory.
        embedding: Embedding vector of the memory.
        metadata: Stored metadata dictionary.
        cluster_id: ID of the cluster containing this memory.
        similarity: Raw cosine similarity to query.
        decay_adjusted_score: Score after temporal decay applied.
        is_representative: True if this is the cluster representative.
        rank: Position in search results (1-based).

    Example:
        >>> results = mcs.search(read_input)
        >>> for result in results:
        ...     print(f"{result.rank}. [{result.similarity:.2f}] {result.content[:50]}...")
    """

    memory_id: str = Field(
        description="Unique memory identifier",
    )
    content: str = Field(
        description="Original text content",
    )
    embedding: list[float] = Field(
        description="Memory embedding vector",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Stored metadata",
    )
    cluster_id: int = Field(
        ge=0,
        description="Cluster containing this memory",
    )
    similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Raw cosine similarity to query",
    )
    decay_adjusted_score: float = Field(
        ge=0.0,
        description="Score after temporal decay",
    )
    is_representative: bool = Field(
        description="True if cluster representative",
    )
    rank: int = Field(
        ge=1,
        description="Position in search results (1-based)",
    )

    model_config = {"frozen": True}


class ClusterInfo(BaseModel):
    """Information about a memory cluster.

    Provides cluster statistics and metadata for monitoring
    and debugging purposes.

    Attributes:
        cluster_id: Unique cluster identifier.
        size: Number of memories in the cluster.
        is_consolidated: True if cluster has been consolidated.
        prototype_norm: L2 norm of the prototype (should be ~1.0).
        creation_timestamp: Unix timestamp of cluster creation.
        last_access_timestamp: Unix timestamp of last access.
        access_count: Number of times cluster was accessed.
        representative_content: Preview of representative content.
    """

    cluster_id: int = Field(
        ge=0,
        description="Unique cluster identifier",
    )
    size: int = Field(
        ge=1,
        description="Number of memories in cluster",
    )
    is_consolidated: bool = Field(
        description="True if cluster has been consolidated",
    )
    prototype_norm: float = Field(
        ge=0.0,
        description="L2 norm of prototype vector",
    )
    creation_timestamp: float = Field(
        ge=0.0,
        description="Unix timestamp of creation",
    )
    last_access_timestamp: float = Field(
        ge=0.0,
        description="Unix timestamp of last access",
    )
    access_count: int = Field(
        ge=0,
        description="Number of accesses",
    )
    representative_content: str = Field(
        max_length=200,
        description="Preview of representative content",
    )

    model_config = {"frozen": True}


class ConsolidationResult(BaseModel):
    """Result of cluster consolidation operation.

    Contains statistics about consolidation performed on
    clusters that exceeded the consolidation threshold.

    Attributes:
        clusters_processed: Number of clusters consolidated.
        memories_consolidated: Total memories merged into prototypes.
        storage_reduction: Estimated storage reduction percentage.
        duration_ms: Time taken for consolidation in milliseconds.
    """

    clusters_processed: int = Field(
        ge=0,
        description="Number of clusters consolidated",
    )
    memories_consolidated: int = Field(
        ge=0,
        description="Total memories merged",
    )
    storage_reduction: float = Field(
        ge=0.0,
        le=100.0,
        description="Storage reduction percentage",
    )
    duration_ms: float = Field(
        ge=0.0,
        description="Consolidation duration in milliseconds",
    )

    model_config = {"frozen": True}
