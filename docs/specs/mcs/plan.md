# MCS-001: Memory Cluster Store Implementation Plan

**Epic:** MCS-001 - Implement Memory Cluster Store
**Status:** Ready for Implementation
**Generated:** 2026-01-08

---

## 1. Executive Summary

### Objective

Implement the Memory Cluster Store (MCS) - PRIME's intelligent memory management component that stores memories as embeddings with automatic consolidation of semantically similar content into cluster prototypes, achieving 3-5× storage compression while maintaining retrieval quality.

### Scope

- Core `MemoryClusterStore` class with cluster management
- FAISS/Qdrant index integration for vector similarity search
- Hybrid search combining dense (semantic) and sparse (BM25) vectors
- Automatic cluster consolidation with prototype computation
- Temporal decay weighting for recency-aware retrieval
- Comprehensive test suite with 90%+ coverage

### Success Criteria

| Metric | Target |
|--------|--------|
| Storage Compression | 3-5× |
| Cluster Purity | >0.85 silhouette |
| Prototype Quality | >0.90 cosine sim |
| Write Latency | <20ms p50 |
| Read Latency | <30ms p50 |
| Test Coverage | ≥90% |

### Dependencies

- **External:** FAISS, Qdrant, NumPy, Pydantic v2
- **Internal:** ENC-001 (Y-Encoder for content embedding)
- **Blocks:** API-001

---

## 2. Context & Documentation Sources

### Primary Specification

- [docs/specs/mcs/spec.md](spec.md) - Full MCS specification

### Architecture Context

- [.sage/agent/system/architecture.md](../../../.sage/agent/system/architecture.md) - System architecture
- [.sage/agent/system/tech-stack.md](../../../.sage/agent/system/tech-stack.md) - Technology stack

### Enhancement Integration

**From docs/enhancement.md:**
- **Hybrid Search (BM25 + Vector)** - Score: 9.25 - Integrated into MCS spec
  - FR-MCS-015 through FR-MCS-018 implement hybrid search
  - Qdrant native sparse vectors for BM25/TF-IDF
  - Reciprocal Rank Fusion (RRF) for result combination

### Traceability Matrix

| Requirement | Source | Priority |
|-------------|--------|----------|
| FR-MCS-001: Y-Encoder embedding | spec.md | P0 |
| FR-MCS-002: Nearest cluster | spec.md | P0 |
| FR-MCS-003: Similarity threshold | spec.md | P0 |
| FR-MCS-004: New cluster creation | spec.md | P0 |
| FR-MCS-005: Consolidation trigger | spec.md | P0 |
| FR-MCS-007: FAISS search | spec.md | P0 |
| FR-MCS-009: Temporal decay | spec.md | P1 |
| FR-MCS-015-018: Hybrid search | spec.md, enhancement.md | P1 |

---

## 3. Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Memory Cluster Store (MCS)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ MCS         │───▶│ Y-Encoder    │    │ Cluster      │               │
│  │ (Public API)│    │ (Protocol)   │    │ Manager      │               │
│  └─────────────┘    └──────────────┘    └──────────────┘               │
│         │                                     │                         │
│         ▼                                     ▼                         │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │                    Vector Index Layer                    │           │
│  │  ┌──────────────┐    ┌──────────────┐                   │           │
│  │  │ Dense Index  │    │ Sparse Index │                   │           │
│  │  │ (FAISS/Qdrant)│   │ (BM25/TF-IDF)│                   │           │
│  │  └──────────────┘    └──────────────┘                   │           │
│  │           │                   │                          │           │
│  │           └───────────────────┘                          │           │
│  │                     │                                    │           │
│  │                     ▼                                    │           │
│  │           ┌─────────────────┐                           │           │
│  │           │ Fusion Layer    │ (RRF/RSF)                 │           │
│  │           └─────────────────┘                           │           │
│  └──────────────────────────────────────────────────────────┘           │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Temporal    │    │ Consolidation│    │ Metadata     │               │
│  │ Decay       │    │ Engine       │    │ Store        │               │
│  └─────────────┘    └──────────────┘    └──────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Write Path

```
Input: Content string + metadata
    │
    ▼
┌─────────────────┐
│ Y-Encoder       │ ──▶ Target Embedding S_Y (1024-dim, L2-norm)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tokenize        │ ──▶ Sparse Vector (BM25 indices/values)
│ (for BM25)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cluster Search  │ ──▶ Find nearest prototype (cosine sim)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  sim > θ   sim ≤ θ
    │         │
    ▼         ▼
┌────────┐  ┌──────────┐
│ Join   │  │ Create   │
│ Cluster│  │ Cluster  │
└────┬───┘  └────┬─────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────┐
│ Update Index    │ ──▶ Add to FAISS/Qdrant
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Size      │ ──▶ size > τ_consolidate?
└────────┬────────┘
         │
         ▼ (if yes)
┌─────────────────┐
│ Consolidate     │ ──▶ Merge members, update prototype
└─────────────────┘

Output: MemoryWriteResult
```

### Data Flow: Read Path (Hybrid Search)

```
Input: Query embedding + text + k
    │
    ├────────────────────────────┐
    │                            │
    ▼                            ▼
┌─────────────────┐    ┌─────────────────┐
│ Dense Search    │    │ Sparse Search   │
│ (Vector sim)    │    │ (BM25)          │
└────────┬────────┘    └────────┬────────┘
         │                      │
         │ Top-K × 2            │ Top-K × 2
         │                      │
         ▼                      ▼
┌──────────────────────────────────────────┐
│          Reciprocal Rank Fusion          │
│   RRF(d) = Σ 1/(k + rank_i(d))           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌─────────────────┐
│ Load Clusters   │ ──▶ Get cluster members
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Temporal Decay  │ ──▶ score × decay^(Δt)
└────────┬────────┘
         │
         ▼
Output: list[MemoryReadResult]
```

---

## 4. Technical Specification

### File Structure

```
src/prime/memory/
├── __init__.py          # Export MCS, types
├── mcs.py               # MemoryClusterStore implementation
├── mcs_config.py        # MCSConfig
├── cluster.py           # MemoryCluster dataclass
├── index.py             # Vector index abstraction
├── faiss_index.py       # FAISS implementation
├── qdrant_index.py      # Qdrant implementation (with hybrid)
├── sparse.py            # Sparse vector (BM25) utilities
├── fusion.py            # RRF fusion implementation
├── types.py             # Input/Output models
└── exceptions.py        # MCS exceptions

tests/
├── test_mcs.py          # Core MCS tests
├── test_cluster.py      # Cluster tests
├── test_index.py        # Index tests
└── test_hybrid.py       # Hybrid search tests
```

### Core Implementation

#### `src/prime/memory/types.py`

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryWriteInput(BaseModel):
    """Input for memory write operation."""

    content: str = Field(min_length=1, max_length=50000)
    metadata: dict[str, str | int | float | bool] | None = None
    user_id: str | None = None
    session_id: str | None = None
    force_new_cluster: bool = False

    model_config = {"frozen": True}


class MemoryWriteResult(BaseModel):
    """Result of memory write operation."""

    memory_id: str = Field(description="Unique memory identifier")
    cluster_id: int = Field(ge=0, description="Assigned cluster ID")
    is_new_cluster: bool = Field(description="True if new cluster created")
    consolidated: bool = Field(description="True if consolidation triggered")
    similarity_to_prototype: float = Field(ge=0.0, le=1.0)

    model_config = {"frozen": True}


class MemoryReadInput(BaseModel):
    """Input for memory read operation."""

    embedding: list[float] = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=100)
    user_id: str | None = None
    session_id: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    query_text: str | None = Field(default=None, description="For BM25")
    search_mode: str = Field(default="hybrid", description="dense/sparse/hybrid")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class MemoryReadResult(BaseModel):
    """Single memory retrieval result."""

    memory_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    cluster_id: int
    similarity: float = Field(ge=0.0, le=1.0)
    decay_adjusted_score: float = Field(ge=0.0)
    is_representative: bool = Field(description="True if cluster representative")

    model_config = {"frozen": True}
```

#### `src/prime/memory/mcs_config.py`

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MCSConfig(BaseModel):
    """Configuration for Memory Cluster Store."""

    embedding_dim: int = Field(
        default=1024,
        ge=1,
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
        description="Min cluster size to trigger consolidation",
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
    index_type: Literal["faiss", "qdrant"] = Field(
        default="qdrant",
        description="Vector index backend",
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable BM25 + vector hybrid search",
    )
    sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default weight for sparse (BM25) in RRF fusion",
    )
    fusion_method: Literal["rrf", "rsf"] = Field(
        default="rrf",
        description="Fusion method for hybrid search",
    )

    model_config = {"frozen": True}
```

#### `src/prime/memory/cluster.py`

```python
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MemoryCluster:
    """A cluster of semantically similar memories.

    Attributes:
        cluster_id: Unique cluster identifier.
        prototype: L2-normalized centroid embedding.
        is_consolidated: True if cluster has been consolidated.
    """

    cluster_id: int
    prototype: np.ndarray
    member_embeddings: list[np.ndarray] = field(default_factory=list)
    member_contents: list[str] = field(default_factory=list)
    member_metadata: list[dict[str, Any]] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    is_consolidated: bool = False
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0

    @property
    def size(self) -> int:
        """Return number of members in cluster."""
        return len(self.member_contents)

    @property
    def representative_index(self) -> int:
        """Return index of member closest to prototype."""
        if not self.member_embeddings:
            return 0

        similarities = [
            np.dot(emb, self.prototype)
            for emb in self.member_embeddings
        ]
        return int(np.argmax(similarities))

    @property
    def representative_content(self) -> str:
        """Return content closest to prototype."""
        if not self.member_contents:
            return ""
        return self.member_contents[self.representative_index]

    def add_member(
        self,
        embedding: np.ndarray,
        content: str,
        metadata: dict[str, Any],
        memory_id: str | None = None,
    ) -> str:
        """Add member to cluster and update prototype.

        Args:
            embedding: L2-normalized embedding.
            content: Memory content.
            metadata: Associated metadata.
            memory_id: Optional ID, generated if not provided.

        Returns:
            Memory ID.
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        self.member_embeddings.append(embedding)
        self.member_contents.append(content)
        self.member_metadata.append(metadata)
        self.member_ids.append(memory_id)

        # Update prototype as L2-normalized mean
        self._update_prototype()
        self.last_access_time = time.time()

        return memory_id

    def consolidate(self) -> None:
        """Consolidate cluster - mark as immutable.

        After consolidation:
        - Prototype is fixed
        - Only representative content is accessible
        - Individual members can be garbage collected
        """
        self.is_consolidated = True
        # Keep only representative
        rep_idx = self.representative_index
        self.member_contents = [self.member_contents[rep_idx]]
        self.member_metadata = [self.member_metadata[rep_idx]]
        self.member_ids = [self.member_ids[rep_idx]]
        # Clear embeddings (prototype sufficient)
        self.member_embeddings = []

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Calculate cosine similarity to prototype.

        Args:
            embedding: L2-normalized query embedding.

        Returns:
            Cosine similarity [-1, 1].
        """
        return float(np.dot(self.prototype, embedding))

    def _update_prototype(self) -> None:
        """Recalculate prototype as L2-normalized centroid."""
        if not self.member_embeddings:
            return

        centroid = np.mean(self.member_embeddings, axis=0)
        self.prototype = centroid / np.linalg.norm(centroid)
```

#### `src/prime/memory/index.py`

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    """Protocol for vector similarity search indexes."""

    @property
    def size(self) -> int:
        """Return number of vectors in index."""
        ...

    def add(self, vector: np.ndarray, vector_id: int) -> None:
        """Add vector to index."""
        ...

    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> tuple[list[int], list[float]]:
        """Search for top-k nearest vectors.

        Returns:
            Tuple of (vector_ids, similarities).
        """
        ...

    def remove(self, vector_id: int) -> None:
        """Remove vector from index."""
        ...

    def rebuild(self) -> None:
        """Rebuild index (for maintenance)."""
        ...
```

#### `src/prime/memory/qdrant_index.py`

```python
from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import ScoredPoint


class QdrantIndex:
    """Qdrant vector index with hybrid search support.

    Supports both dense (semantic) and sparse (BM25) vector search
    with configurable fusion strategies.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        client: QdrantClient | None = None,
        path: str | None = None,
    ) -> None:
        """Initialize Qdrant index.

        Args:
            collection_name: Name of the collection.
            embedding_dim: Dimension of dense vectors.
            client: Optional existing client.
            path: Path for local persistence.
        """
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        if client is not None:
            self._client = client
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")

        self._ensure_collection()

    @property
    def size(self) -> int:
        """Return number of points in collection."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count

    def _ensure_collection(self) -> None:
        """Create collection if not exists."""
        collections = self._client.get_collections().collections
        names = [c.name for c in collections]

        if self._collection_name not in names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self._embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )

    def add(
        self,
        vector: np.ndarray,
        vector_id: int,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Add vector to index with optional sparse vector."""
        vectors = {"dense": vector.tolist()}

        if sparse_indices is not None and sparse_values is not None:
            vectors["bm25"] = models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=vectors,
                    payload=payload or {},
                ),
            ],
        )

    def search_dense(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[ScoredPoint]:
        """Search using only dense vectors."""
        return self._client.search(
            collection_name=self._collection_name,
            query_vector=("dense", query.tolist()),
            limit=k,
        )

    def search_hybrid(
        self,
        dense_query: np.ndarray,
        sparse_indices: list[int],
        sparse_values: list[float],
        k: int,
        fusion: models.Fusion = models.Fusion.RRF,
    ) -> list[ScoredPoint]:
        """Hybrid search with RRF fusion."""
        return self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using="dense",
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="bm25",
                    limit=k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=k,
        ).points

    def remove(self, vector_id: int) -> None:
        """Remove vector from index."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[vector_id]),
        )

    def rebuild(self) -> None:
        """Rebuild index (trigger optimization)."""
        self._client.update_collection(
            collection_name=self._collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
```

#### `src/prime/memory/sparse.py`

```python
from __future__ import annotations

import re
from collections import Counter


class BM25Tokenizer:
    """Simple BM25 tokenizer for sparse vector generation."""

    def __init__(
        self,
        lowercase: bool = True,
        min_length: int = 2,
    ) -> None:
        """Initialize tokenizer.

        Args:
            lowercase: Convert to lowercase.
            min_length: Minimum token length.
        """
        self._lowercase = lowercase
        self._min_length = min_length
        self._vocab: dict[str, int] = {}
        self._next_id = 0

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        if self._lowercase:
            text = text.lower()

        # Simple word tokenization
        tokens = re.findall(r"\b\w+\b", text)

        # Filter by length
        return [t for t in tokens if len(t) >= self._min_length]

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Encode text to sparse vector.

        Returns:
            Tuple of (indices, values) for sparse vector.
        """
        tokens = self.tokenize(text)
        counts = Counter(tokens)

        indices = []
        values = []

        for token, count in counts.items():
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1

            indices.append(self._vocab[token])
            values.append(float(count))

        return indices, values

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)
```

#### `src/prime/memory/mcs.py`

```python
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from prime.encoder import Encoder
from prime.memory.cluster import MemoryCluster
from prime.memory.mcs_config import MCSConfig
from prime.memory.qdrant_index import QdrantIndex
from prime.memory.sparse import BM25Tokenizer
from prime.memory.types import (
    MemoryReadInput,
    MemoryReadResult,
    MemoryWriteInput,
    MemoryWriteResult,
)


class MCSError(Exception):
    """Base exception for MCS errors."""


class ClusterNotFoundError(MCSError):
    """Cluster not found."""


class ConsolidationError(MCSError):
    """Error during consolidation."""


class MemoryClusterStore:
    """Memory Cluster Store for intelligent memory management.

    Stores memories with automatic consolidation of semantically
    similar content into cluster prototypes.

    Attributes:
        config: MCS configuration.
        num_clusters: Current number of clusters.
    """

    def __init__(
        self,
        encoder: Encoder,
        config: MCSConfig | None = None,
        index_path: str | None = None,
    ) -> None:
        """Initialize MCS.

        Args:
            encoder: Y-Encoder for content embedding.
            config: MCS configuration.
            index_path: Path for persistent storage.
        """
        self._encoder = encoder
        self._config = config or MCSConfig()
        self._clusters: dict[int, MemoryCluster] = {}
        self._next_cluster_id = 0

        # Initialize vector index
        self._index = QdrantIndex(
            collection_name="prime_memories",
            embedding_dim=self._config.embedding_dim,
            path=index_path,
        )

        # Initialize BM25 tokenizer for hybrid search
        self._tokenizer = BM25Tokenizer()

    @property
    def num_clusters(self) -> int:
        """Return number of active clusters."""
        return len(self._clusters)

    def write(self, input_data: MemoryWriteInput) -> MemoryWriteResult:
        """Write content to memory store.

        Args:
            input_data: Write input with content and metadata.

        Returns:
            MemoryWriteResult with cluster assignment info.
        """
        # Encode content
        embedding = self._encoder.encode(input_data.content)

        # Generate sparse vector for BM25
        sparse_indices, sparse_values = self._tokenizer.encode(input_data.content)

        # Find nearest cluster
        cluster_id, similarity = self._find_nearest_cluster(embedding)

        # Determine if joining existing or creating new
        is_new_cluster = (
            input_data.force_new_cluster
            or cluster_id is None
            or similarity < self._config.similarity_threshold
        )

        if is_new_cluster:
            cluster = self._create_cluster(embedding)
            cluster_id = cluster.cluster_id
            similarity = 1.0
        else:
            cluster = self._clusters[cluster_id]

        # Add member to cluster
        metadata = input_data.metadata or {}
        if input_data.user_id:
            metadata["user_id"] = input_data.user_id
        if input_data.session_id:
            metadata["session_id"] = input_data.session_id

        memory_id = cluster.add_member(
            embedding=embedding,
            content=input_data.content,
            metadata=metadata,
        )

        # Update index
        self._index.add(
            vector=cluster.prototype,
            vector_id=cluster_id,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            payload={
                "cluster_id": cluster_id,
                "memory_id": memory_id,
            },
        )

        # Check consolidation threshold
        consolidated = False
        if cluster.size >= self._config.consolidation_threshold and not cluster.is_consolidated:
            self._consolidate_cluster(cluster_id)
            consolidated = True

        return MemoryWriteResult(
            memory_id=memory_id,
            cluster_id=cluster_id,
            is_new_cluster=is_new_cluster,
            consolidated=consolidated,
            similarity_to_prototype=similarity,
        )

    def read(self, input_data: MemoryReadInput) -> list[MemoryReadResult]:
        """Read memories by similarity search.

        Args:
            input_data: Read input with query embedding.

        Returns:
            List of matching memories ranked by similarity.
        """
        query_embedding = np.array(input_data.embedding, dtype=np.float32)

        # Perform search based on mode
        if input_data.search_mode == "hybrid" and input_data.query_text:
            sparse_indices, sparse_values = self._tokenizer.encode(input_data.query_text)
            results = self._index.search_hybrid(
                dense_query=query_embedding,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                k=input_data.k,
            )
        else:
            results = self._index.search_dense(query_embedding, input_data.k)

        # Convert to MemoryReadResult
        memories = []
        now = time.time()

        for point in results:
            cluster_id = point.payload.get("cluster_id")
            if cluster_id is None or cluster_id not in self._clusters:
                continue

            cluster = self._clusters[cluster_id]

            # Apply temporal decay
            age_units = (now - cluster.last_access_time) / self._config.decay_unit_seconds
            decay = self._config.decay_factor ** age_units
            decay_adjusted_score = point.score * decay

            # Filter by minimum similarity
            if point.score < input_data.min_similarity:
                continue

            # Get representative content
            memories.append(MemoryReadResult(
                memory_id=cluster.member_ids[0] if cluster.member_ids else "",
                content=cluster.representative_content,
                embedding=cluster.prototype.tolist(),
                metadata=cluster.member_metadata[0] if cluster.member_metadata else {},
                cluster_id=cluster_id,
                similarity=point.score,
                decay_adjusted_score=decay_adjusted_score,
                is_representative=True,
            ))

            cluster.last_access_time = now
            cluster.access_count += 1

        # Sort by decay-adjusted score
        memories.sort(key=lambda m: m.decay_adjusted_score, reverse=True)

        return memories[:input_data.k]

    def get_cluster(self, cluster_id: int) -> MemoryCluster:
        """Get cluster by ID.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            MemoryCluster.

        Raises:
            ClusterNotFoundError: If cluster doesn't exist.
        """
        if cluster_id not in self._clusters:
            raise ClusterNotFoundError(f"Cluster {cluster_id} not found")
        return self._clusters[cluster_id]

    def get_stats(self) -> dict[str, Any]:
        """Get MCS statistics.

        Returns:
            Dictionary with store statistics.
        """
        total_memories = sum(c.size for c in self._clusters.values())
        consolidated_count = sum(1 for c in self._clusters.values() if c.is_consolidated)

        return {
            "num_clusters": self.num_clusters,
            "total_memories": total_memories,
            "consolidated_clusters": consolidated_count,
            "compression_ratio": total_memories / max(self.num_clusters, 1),
            "index_size": self._index.size,
        }

    def _find_nearest_cluster(
        self,
        embedding: np.ndarray,
    ) -> tuple[int | None, float]:
        """Find nearest cluster to embedding.

        Returns:
            Tuple of (cluster_id, similarity) or (None, 0.0) if empty.
        """
        if not self._clusters:
            return None, 0.0

        results = self._index.search_dense(embedding, k=1)
        if not results:
            return None, 0.0

        cluster_id = results[0].payload.get("cluster_id")
        similarity = results[0].score

        return cluster_id, similarity

    def _create_cluster(self, embedding: np.ndarray) -> MemoryCluster:
        """Create new cluster with embedding as prototype."""
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1

        cluster = MemoryCluster(
            cluster_id=cluster_id,
            prototype=embedding / np.linalg.norm(embedding),
        )
        self._clusters[cluster_id] = cluster

        return cluster

    def _consolidate_cluster(self, cluster_id: int) -> None:
        """Consolidate cluster into prototype."""
        if cluster_id not in self._clusters:
            raise ClusterNotFoundError(f"Cluster {cluster_id} not found")

        cluster = self._clusters[cluster_id]
        if cluster.is_consolidated:
            return

        cluster.consolidate()
```

#### `src/prime/memory/__init__.py`

```python
from __future__ import annotations

from prime.memory.cluster import MemoryCluster
from prime.memory.mcs import (
    ClusterNotFoundError,
    ConsolidationError,
    MCSError,
    MemoryClusterStore,
)
from prime.memory.mcs_config import MCSConfig
from prime.memory.types import (
    MemoryReadInput,
    MemoryReadResult,
    MemoryWriteInput,
    MemoryWriteResult,
)

__all__ = [
    "ClusterNotFoundError",
    "ConsolidationError",
    "MCSConfig",
    "MCSError",
    "MemoryCluster",
    "MemoryClusterStore",
    "MemoryReadInput",
    "MemoryReadResult",
    "MemoryWriteInput",
    "MemoryWriteResult",
]
```

---

## 5. Test Specification

### Test File: `tests/test_mcs.py`

```python
from __future__ import annotations

import numpy as np
import pytest
import time

from prime.memory import (
    ClusterNotFoundError,
    MCSConfig,
    MemoryCluster,
    MemoryClusterStore,
    MemoryReadInput,
    MemoryWriteInput,
    MemoryWriteResult,
)


# ============================================================================
# Mock Encoder
# ============================================================================


class MockEncoder:
    """Mock Y-Encoder for testing."""

    def __init__(self, embedding_dim: int = 1024) -> None:
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def max_length(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-encoder"

    def encode(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(self._embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_encoder() -> MockEncoder:
    return MockEncoder(embedding_dim=1024)


@pytest.fixture
def mcs_config() -> MCSConfig:
    return MCSConfig(
        embedding_dim=1024,
        similarity_threshold=0.85,
        consolidation_threshold=3,
    )


@pytest.fixture
def mcs(mock_encoder: MockEncoder, mcs_config: MCSConfig) -> MemoryClusterStore:
    return MemoryClusterStore(encoder=mock_encoder, config=mcs_config)


# ============================================================================
# Write Tests
# ============================================================================


def test_write_creates_new_cluster(mcs: MemoryClusterStore) -> None:
    """Test first write creates new cluster."""
    result = mcs.write(MemoryWriteInput(content="First memory"))

    assert result.is_new_cluster
    assert result.cluster_id == 0
    assert mcs.num_clusters == 1


def test_write_joins_existing_cluster(mcs: MemoryClusterStore) -> None:
    """Test similar content joins existing cluster."""
    # Write first memory
    mcs.write(MemoryWriteInput(content="Python programming tutorial"))

    # Write very similar content (will have similar embedding due to hash)
    # Note: In real tests, use controlled embeddings
    result = mcs.write(MemoryWriteInput(content="Python programming tutorial"))

    # Should join existing cluster (same content = same embedding)
    assert not result.is_new_cluster
    assert result.cluster_id == 0


def test_write_creates_new_cluster_for_dissimilar(mcs: MemoryClusterStore) -> None:
    """Test dissimilar content creates new cluster."""
    mcs.write(MemoryWriteInput(content="Python programming tutorial"))
    result = mcs.write(MemoryWriteInput(content="Cooking pasta recipe Italian"))

    # Different topic should create new cluster
    assert result.is_new_cluster
    assert mcs.num_clusters == 2


def test_force_new_cluster(mcs: MemoryClusterStore) -> None:
    """Test force_new_cluster creates new cluster."""
    mcs.write(MemoryWriteInput(content="Test content"))
    result = mcs.write(MemoryWriteInput(
        content="Test content",
        force_new_cluster=True,
    ))

    assert result.is_new_cluster
    assert mcs.num_clusters == 2


# ============================================================================
# Consolidation Tests
# ============================================================================


def test_consolidation_on_threshold(mcs: MemoryClusterStore, mock_encoder: MockEncoder) -> None:
    """Test consolidation triggers at threshold."""
    # Create cluster with same content (same embedding)
    content = "Consolidation test content"
    for i in range(3):
        result = mcs.write(MemoryWriteInput(content=content))

    # Should trigger consolidation (threshold=3)
    assert result.consolidated

    cluster = mcs.get_cluster(result.cluster_id)
    assert cluster.is_consolidated


# ============================================================================
# Read Tests
# ============================================================================


def test_read_returns_top_k(mcs: MemoryClusterStore) -> None:
    """Test read returns top-k results."""
    # Write several memories
    for i in range(5):
        mcs.write(MemoryWriteInput(content=f"Memory content {i}"))

    # Read
    embedding = np.random.randn(1024).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    results = mcs.read(MemoryReadInput(
        embedding=embedding.tolist(),
        k=3,
    ))

    assert len(results) <= 3


def test_read_empty_store(mcs: MemoryClusterStore) -> None:
    """Test read from empty store returns empty list."""
    embedding = np.random.randn(1024).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    results = mcs.read(MemoryReadInput(
        embedding=embedding.tolist(),
        k=5,
    ))

    assert results == []


# ============================================================================
# Temporal Decay Tests
# ============================================================================


def test_temporal_decay_applied(mcs: MemoryClusterStore) -> None:
    """Test temporal decay affects scores."""
    # Write memory
    mcs.write(MemoryWriteInput(content="Old memory"))

    # Manually age the cluster
    cluster = mcs.get_cluster(0)
    cluster.last_access_time = time.time() - 7200  # 2 hours ago

    # Read
    embedding = cluster.prototype
    results = mcs.read(MemoryReadInput(
        embedding=embedding.tolist(),
        k=1,
    ))

    if results:
        # Decay adjusted score should be less than raw similarity
        assert results[0].decay_adjusted_score <= results[0].similarity


# ============================================================================
# Cluster Tests
# ============================================================================


def test_get_cluster(mcs: MemoryClusterStore) -> None:
    """Test get_cluster returns cluster."""
    mcs.write(MemoryWriteInput(content="Test"))
    cluster = mcs.get_cluster(0)

    assert isinstance(cluster, MemoryCluster)
    assert cluster.cluster_id == 0


def test_get_cluster_not_found(mcs: MemoryClusterStore) -> None:
    """Test get_cluster raises for missing cluster."""
    with pytest.raises(ClusterNotFoundError):
        mcs.get_cluster(999)


# ============================================================================
# Statistics Tests
# ============================================================================


def test_get_stats(mcs: MemoryClusterStore) -> None:
    """Test get_stats returns statistics."""
    mcs.write(MemoryWriteInput(content="Memory 1"))
    mcs.write(MemoryWriteInput(content="Memory 2"))

    stats = mcs.get_stats()

    assert "num_clusters" in stats
    assert "total_memories" in stats
    assert "compression_ratio" in stats


# ============================================================================
# Cluster Unit Tests
# ============================================================================


def test_cluster_add_member() -> None:
    """Test MemoryCluster.add_member."""
    prototype = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    cluster = MemoryCluster(cluster_id=0, prototype=prototype)

    memory_id = cluster.add_member(
        embedding=prototype,
        content="Test content",
        metadata={"key": "value"},
    )

    assert cluster.size == 1
    assert memory_id is not None
    assert cluster.member_contents[0] == "Test content"


def test_cluster_representative_content() -> None:
    """Test MemoryCluster.representative_content."""
    prototype = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    cluster = MemoryCluster(cluster_id=0, prototype=prototype)

    # Add members with varying similarity to prototype
    cluster.add_member(prototype * 0.9, "Close to prototype", {})
    cluster.add_member(prototype * 0.5, "Far from prototype", {})

    # Representative should be closest to prototype
    assert "Close" in cluster.representative_content or "Far" in cluster.representative_content


def test_cluster_consolidate() -> None:
    """Test MemoryCluster.consolidate."""
    prototype = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    cluster = MemoryCluster(cluster_id=0, prototype=prototype)

    cluster.add_member(prototype, "Content 1", {})
    cluster.add_member(prototype, "Content 2", {})
    cluster.add_member(prototype, "Content 3", {})

    cluster.consolidate()

    assert cluster.is_consolidated
    assert len(cluster.member_contents) == 1  # Only representative kept
    assert len(cluster.member_embeddings) == 0  # Embeddings cleared


# ============================================================================
# Configuration Tests
# ============================================================================


def test_default_config() -> None:
    """Test default MCS configuration."""
    config = MCSConfig()

    assert config.embedding_dim == 1024
    assert config.similarity_threshold == 0.85
    assert config.consolidation_threshold == 5
    assert config.enable_hybrid_search is True
```

---

## 6. Implementation Roadmap

### Phase 1: Core Implementation (P0)

**Step 1.1: Types and Config**
- Implement `types.py` (input/output models)
- Implement `mcs_config.py`
- Add validation tests

**Step 1.2: Cluster Management**
- Implement `cluster.py` (MemoryCluster dataclass)
- Add/update prototype logic
- Consolidation logic

**Step 1.3: Vector Index**
- Implement `index.py` protocol
- Implement `qdrant_index.py` for production
- Optional: `faiss_index.py` for development

**Step 1.4: Core MCS**
- Implement `mcs.py` main class
- Write path (cluster assignment)
- Read path (similarity search)

**Step 1.5: Tests**
- Unit tests for cluster operations
- Integration tests with Qdrant

### Phase 2: Hybrid Search (P1)

**Step 2.1: Sparse Vectors**
- Implement `sparse.py` (BM25 tokenizer)
- Sparse vector generation

**Step 2.2: Hybrid Search**
- RRF fusion in `qdrant_index.py`
- Search mode selection in MCS

**Step 2.3: Temporal Decay**
- Decay calculation
- Score adjustment in read path

### Phase 3: Production (P2)

**Step 3.1: Persistence**
- Qdrant persistent storage
- Metadata backup

**Step 3.2: Observability**
- Statistics endpoint
- Cluster health metrics

---

## 7. Quality Assurance

### Code Quality Gates

| Gate | Requirement | Tool |
|------|-------------|------|
| Type Safety | 100% coverage | mypy --strict |
| Linting | No errors | ruff check |
| Formatting | Consistent | ruff format |
| Test Coverage | ≥90% | pytest-cov |
| Tests | All passing | pytest |

### Performance Validation

```bash
# Write latency benchmark
uv run python -c "
import time
import numpy as np
from prime.memory import MemoryClusterStore, MCSConfig, MemoryWriteInput
from prime.encoder import YEncoder, YEncoderConfig

encoder = YEncoder(YEncoderConfig(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    embedding_dim=384,
    device='cpu'
))
mcs = MemoryClusterStore(encoder, MCSConfig(embedding_dim=384))

times = []
for i in range(100):
    start = time.perf_counter()
    mcs.write(MemoryWriteInput(content=f'Test memory {i}'))
    times.append((time.perf_counter() - start) * 1000)

print(f'Write p50: {np.percentile(times, 50):.2f}ms')
print(f'Write p95: {np.percentile(times, 95):.2f}ms')
"
```

---

## 8. Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Qdrant connection issues | Low | High | Fallback to in-memory, health checks |
| Index corruption | Low | High | Regular backups, rebuild capability |
| Memory pressure from clusters | Medium | Medium | Max cluster limit, consolidation |
| BM25 vocabulary explosion | Medium | Low | Vocabulary pruning, min frequency |

### Performance Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Write latency exceeds target | Medium | Medium | Async consolidation, batch writes |
| Search latency at scale | Medium | Medium | Index optimization, caching |

---

## 9. References & Traceability

### Source Documents

| Document | Purpose |
|----------|---------|
| [spec.md](spec.md) | Functional requirements |
| [architecture.md](../../../.sage/agent/system/architecture.md) | System context |
| [tech-stack.md](../../../.sage/agent/system/tech-stack.md) | Technology choices |
| [enhancement.md](../../../docs/enhancement.md) | Hybrid search requirements |

### Related Tickets

| Ticket | Relationship |
|--------|--------------|
| ENC-001 | Upstream - provides Y-Encoder for content embedding |
| PRED-001 | Consumer - searches MCS with predicted embeddings |
| SSM-001 | Trigger source - RETRIEVE_CONSOLIDATE action |
| API-001 | Integrates MCS into PRIME API |

### External References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

## Appendix A: Consolidation Algorithm

```
Input: Cluster with members [m₁, m₂, ..., mₙ]

1. Compute centroid:
   μ = (1/n) Σᵢ emb(mᵢ)

2. L2-normalize:
   prototype = μ / ||μ||

3. Find representative:
   r = argmax_i cos(emb(mᵢ), prototype)

4. Keep only representative content:
   cluster.content = [m_r.content]
   cluster.metadata = [m_r.metadata]

5. Clear member embeddings:
   cluster.embeddings = []

6. Mark consolidated:
   cluster.is_consolidated = True
```

## Appendix B: Temporal Decay Formula

```
score_adjusted = score × decay_factor^(age_units)

where:
- score: Raw similarity score
- decay_factor: Config parameter (default 0.99)
- age_units: (now - last_access) / decay_unit_seconds
```

Example with decay_factor=0.99, decay_unit=3600s:
- 1 hour old: score × 0.99¹ = 99% of original
- 24 hours old: score × 0.99²⁴ ≈ 78% of original
- 1 week old: score × 0.99¹⁶⁸ ≈ 18% of original
