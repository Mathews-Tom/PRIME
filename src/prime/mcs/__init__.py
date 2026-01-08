"""Memory Cluster Store (MCS) module.

Provides intelligent memory management with automatic consolidation
of semantically similar content into cluster prototypes. Features
hybrid search (BM25 + vector) with RRF fusion for improved retrieval.

Key Components:
    MemoryClusterStore: Main class for memory operations
    MCSConfig: Configuration schema with validation
    MemoryCluster: Cluster data structure with prototype management

Example:
    >>> from prime.mcs import MemoryClusterStore, MCSConfig
    >>> config = MCSConfig(similarity_threshold=0.85)
    >>> mcs = MemoryClusterStore(config=config, encoder=y_encoder)
    >>> result = mcs.write(content="User prefers dark mode")
    >>> memories = mcs.search(query_embedding, k=5)
"""

from __future__ import annotations

from prime.mcs.cluster import ClusterMember, MemoryCluster
from prime.mcs.exceptions import (
    ClusterNotFoundError,
    ConfigurationError,
    ConsolidationError,
    IndexError,
    MCSError,
    SearchError,
    WriteError,
)
from prime.mcs.index import IndexSearchResult, SparseVector, VectorIndex
from prime.mcs.mcs_config import (
    COMPACT_CONFIG,
    DEFAULT_CONFIG,
    HIGH_PRECISION_CONFIG,
    MEMORY_EFFICIENT_CONFIG,
    MCSConfig,
)
from prime.mcs.qdrant_index import QdrantIndex
from prime.mcs.types import (
    ClusterInfo,
    ConsolidationResult,
    MemoryReadInput,
    MemoryReadResult,
    MemoryWriteInput,
    MemoryWriteResult,
    SearchMode,
)

__all__ = [
    "COMPACT_CONFIG",
    "ClusterInfo",
    "ClusterMember",
    "ClusterNotFoundError",
    "ConfigurationError",
    "ConsolidationError",
    "ConsolidationResult",
    "DEFAULT_CONFIG",
    "HIGH_PRECISION_CONFIG",
    "IndexError",
    "IndexSearchResult",
    "MCSConfig",
    "MCSError",
    "MEMORY_EFFICIENT_CONFIG",
    "MemoryCluster",
    "MemoryReadInput",
    "MemoryReadResult",
    "MemoryWriteInput",
    "MemoryWriteResult",
    "QdrantIndex",
    "SearchError",
    "SearchMode",
    "SparseVector",
    "VectorIndex",
    "WriteError",
]
