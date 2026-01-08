"""Configuration schema for Memory Cluster Store.

Defines immutable configuration using Pydantic v2 with validation
for all MCS parameters including clustering, search, and decay settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MCSConfig(BaseModel):
    """Configuration for Memory Cluster Store.

    Immutable configuration schema with validation for MCS setup.
    Controls clustering behavior, hybrid search parameters, and
    temporal decay settings.

    Attributes:
        embedding_dim: Dimension of stored embeddings (must match Y-Encoder).
        similarity_threshold: Cosine similarity threshold for cluster membership.
            Memories with similarity > threshold join existing clusters.
        consolidation_threshold: Minimum cluster size to trigger consolidation.
            Clusters exceeding this size have members merged into prototype.
        max_clusters: Maximum number of clusters allowed in the store.
        decay_factor: Temporal decay factor per time unit (0.99 = 1% decay).
        decay_unit_seconds: Time unit for decay calculation in seconds.
        index_type: Vector index backend type.
        collection_name: Name of the Qdrant collection for storage.
        enable_hybrid_search: Enable BM25 + vector hybrid search.
        sparse_weight: Weight for sparse (BM25) vectors in RRF fusion.
        fusion_k: RRF fusion constant (higher = more weight to top ranks).
        prefetch_multiplier: Multiplier for prefetch size in hybrid search.
        qdrant_host: Qdrant server hostname.
        qdrant_port: Qdrant server port.
        qdrant_api_key: Optional API key for Qdrant cloud.
        qdrant_prefer_grpc: Use gRPC instead of REST for Qdrant.
    """

    embedding_dim: int = Field(
        default=1024,
        gt=0,
        description="Dimension of stored embeddings",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cluster membership",
    )
    consolidation_threshold: int = Field(
        default=5,
        ge=2,
        description="Minimum cluster size to trigger consolidation",
    )
    max_clusters: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of clusters",
    )
    decay_factor: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Temporal decay factor per time unit",
    )
    decay_unit_seconds: int = Field(
        default=3600,
        ge=1,
        description="Time unit for decay calculation (seconds)",
    )
    index_type: Literal["qdrant", "faiss"] = Field(
        default="qdrant",
        description="Vector index type: 'qdrant' or 'faiss'",
    )
    collection_name: str = Field(
        default="prime_memories",
        min_length=1,
        description="Qdrant collection name",
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable BM25 + vector hybrid search",
    )
    sparse_vocab_size: int = Field(
        default=30000,
        ge=1000,
        le=100000,
        description="Maximum vocabulary size for BM25 sparse vectors",
    )
    sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for sparse (BM25) in RRF fusion",
    )
    fusion_k: int = Field(
        default=60,
        ge=1,
        description="RRF fusion constant k (score = 1/(k + rank))",
    )
    prefetch_multiplier: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Multiplier for prefetch size in hybrid search",
    )
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server hostname",
    )
    qdrant_port: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Qdrant server port",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional API key for Qdrant cloud",
    )
    qdrant_prefer_grpc: bool = Field(
        default=True,
        description="Use gRPC instead of REST for Qdrant",
    )

    model_config = {"frozen": True}


# Default configurations for different use cases
DEFAULT_CONFIG = MCSConfig()

COMPACT_CONFIG = MCSConfig(
    similarity_threshold=0.90,
    consolidation_threshold=3,
    max_clusters=1000,
    decay_factor=0.95,
)

HIGH_PRECISION_CONFIG = MCSConfig(
    similarity_threshold=0.80,
    consolidation_threshold=10,
    max_clusters=50000,
    enable_hybrid_search=True,
    sparse_weight=0.4,
)

MEMORY_EFFICIENT_CONFIG = MCSConfig(
    embedding_dim=384,
    similarity_threshold=0.88,
    consolidation_threshold=3,
    max_clusters=5000,
    enable_hybrid_search=False,
)
