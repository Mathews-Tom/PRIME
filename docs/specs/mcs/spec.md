# Memory Cluster Store (MCS) Specification

## 1. Overview

### Purpose and Business Value

The Memory Cluster Store (MCS) is PRIME's intelligent memory management component. It stores memories as embeddings with **automatic consolidation of semantically similar content** into cluster prototypes, achieving 3-5× storage compression while maintaining retrieval quality.

**Business Value:**
- 3-5× memory storage compression through consolidation
- Faster retrieval via cluster prototype search (hierarchical)
- Reduced memory fragmentation for long-running sessions
- Temporal decay ensures recent memories are prioritized

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Storage Compression | 3-5× | Total memories / active clusters |
| Cluster Purity | >0.85 | Silhouette score |
| Prototype Quality | >0.90 | Avg cosine similarity to members |
| Retrieval Latency | <50ms | p50 FAISS search time |
| Write Latency | <20ms | p50 cluster assignment time |

### Target Users

- PRIME core system (internal API)
- Embedding Predictor (search target)
- Knowledge ingestion pipelines
- Memory export/audit tools

---

## 2. Functional Requirements

### Core Capabilities

**Write Path:**
- **FR-MCS-001**: The system shall encode content using Y-Encoder to produce target embeddings.
- **FR-MCS-002**: The system shall find the nearest cluster prototype using cosine similarity.
- **FR-MCS-003**: The system shall assign memory to existing cluster if similarity > θ_cluster (default 0.85).
- **FR-MCS-004**: The system shall create new cluster if no similar cluster exists.
- **FR-MCS-005**: The system shall trigger consolidation when cluster size exceeds τ_consolidate (default 5).
- **FR-MCS-006**: The system shall update cluster prototype after each member addition.

**Read Path:**
- **FR-MCS-007**: The system shall search FAISS index using predicted/query embedding.
- **FR-MCS-008**: The system shall return top-K clusters ranked by similarity.
- **FR-MCS-009**: The system shall apply temporal decay weighting to results.
- **FR-MCS-010**: The system shall return cluster members with metadata.

**Hybrid Search (BM25 + Vector):**
- **FR-MCS-015**: The system shall support sparse vector storage for BM25/TF-IDF indexing.
- **FR-MCS-016**: The system shall perform hybrid search combining dense (semantic) and sparse (keyword) vectors.
- **FR-MCS-017**: The system shall use Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
- **FR-MCS-018**: The system shall allow configurable weighting between semantic and keyword search.

**Consolidation:**
- **FR-MCS-011**: The system shall compute consolidated prototype as L2-normalized centroid.
- **FR-MCS-012**: The system shall select representative content (closest to prototype).
- **FR-MCS-013**: The system shall rebuild FAISS index after consolidation.
- **FR-MCS-014**: The system shall mark clusters as consolidated (immutable members).

### User Stories

- **US-MCS-001**: As the PRIME orchestrator, I want to store conversation responses so that they can be retrieved later.
- **US-MCS-002**: As the Predictor, I want to search memories using predicted embeddings so that I can find relevant context.
- **US-MCS-003**: As a developer, I want memories to auto-consolidate so that storage remains efficient.
- **US-MCS-004**: As an operator, I want to view cluster statistics so that I can monitor memory health.
- **US-MCS-005**: As the ingestion pipeline, I want to batch-write documents so that I can efficiently populate memory.

### Business Rules and Constraints

- **BR-MCS-001**: Cluster membership MUST be determined by cosine similarity (not L2 distance).
- **BR-MCS-002**: Prototype MUST be L2-normalized (unit vector).
- **BR-MCS-003**: Consolidation MUST preserve at least representative content for retrieval.
- **BR-MCS-004**: Temporal decay MUST use exponential formula: `score * decay_factor^(now - last_access)`.
- **BR-MCS-005**: FAISS index MUST use IndexFlatIP (inner product for cosine similarity on L2-normed vectors).

---

## 3. Non-Functional Requirements

### Performance Targets

| Metric | Target | Constraint |
|--------|--------|------------|
| Write Latency | <20ms p50 | <50ms p95 |
| Read Latency (top-10) | <30ms p50 | <80ms p95 |
| Consolidation Time | <100ms | Per cluster |
| Memory per 1M vectors | <1.5GB | FAISS index |
| Max Clusters | 100,000 | Default limit |

### Security Requirements

- **SEC-MCS-001**: Content MUST support encryption at rest (defer to storage layer).
- **SEC-MCS-002**: Metadata MUST NOT contain PII without explicit consent flag.
- **SEC-MCS-003**: Cluster access MUST be scoped by user/session ID.

### Scalability Considerations

- Horizontal scaling via cluster ID sharding
- FAISS → Qdrant migration path for >10M vectors
- Background consolidation jobs (async, non-blocking)

---

## 4. Features & Flows

### Feature Breakdown

| Feature | Priority | Description |
|---------|----------|-------------|
| Memory Write | P0 | Store content with embedding and metadata |
| Memory Read | P0 | Retrieve top-K memories by similarity |
| Cluster Assignment | P0 | Auto-assign to nearest cluster |
| Consolidation | P0 | Merge cluster members into prototype |
| Temporal Decay | P1 | Recency-weighted retrieval |
| Batch Write | P1 | Efficient multi-document ingestion |
| **Hybrid Search** | P1 | BM25 + Vector search with RRF fusion |
| Cluster Statistics | P2 | Monitoring and observability |
| Index Rebuild | P2 | Manual index maintenance |

### Key User Flows

**Flow 1: Write Memory**

```
Input: Content string + metadata dict
  │
  ▼
┌─────────────────┐
│ Y-Encoder       │ ──▶ Target Embedding (1024-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cluster Search  │ ──▶ Find nearest prototype
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Similarity Check│ ──▶ sim > θ_cluster?
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   Yes        No
    │         │
    ▼         ▼
┌────────┐  ┌────────────┐
│ Join   │  │ Create New │
│ Cluster│  │ Cluster    │
└────┬───┘  └─────┬──────┘
     │            │
     ▼            ▼
┌─────────────────────┐
│ Check Consolidation │ ──▶ size > τ?
└────────┬────────────┘
         │
         ▼ (if yes)
┌─────────────────┐
│ Consolidate     │
│ Cluster         │
└─────────────────┘

Output: MemoryWriteResult {
  memory_id: str,
  cluster_id: int,
  is_new_cluster: bool,
  consolidated: bool
}
```

**Flow 2: Read Memory**

```
Input: Query/Predicted Embedding + k
  │
  ▼
┌─────────────────┐
│ FAISS Search    │ ──▶ Top-K cluster prototypes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Clusters   │ ──▶ Get cluster members
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply Decay     │ ──▶ Recency weighting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rank & Return   │ ──▶ Sorted memories
└─────────────────┘

Output: list[MemoryReadResult] {
  content: str,
  embedding: ndarray,
  metadata: dict,
  cluster_id: int,
  similarity: float,
  decay_adjusted_score: float
}
```

### Input/Output Specifications

**Write Input:**
```python
class MemoryWriteInput(BaseModel):
    """Input for memory write operation."""

    content: str = Field(min_length=1, max_length=10000)
    metadata: dict[str, str | int | float | bool] | None = None
    user_id: str | None = None
    session_id: str | None = None
    force_new_cluster: bool = False
```

**Write Output:**
```python
class MemoryWriteResult(BaseModel):
    """Result of memory write operation."""

    memory_id: str = Field(description="Unique memory identifier")
    cluster_id: int = Field(ge=0, description="Assigned cluster ID")
    is_new_cluster: bool = Field(description="True if new cluster created")
    consolidated: bool = Field(description="True if consolidation triggered")
    similarity_to_prototype: float = Field(ge=0.0, le=1.0)
```

**Read Input:**
```python
class MemoryReadInput(BaseModel):
    """Input for memory read operation."""

    embedding: list[float] = Field(min_length=1024, max_length=1024)
    k: int = Field(default=5, ge=1, le=100)
    user_id: str | None = None
    session_id: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Hybrid search options
    query_text: str | None = Field(default=None, description="Original text for BM25")
    search_mode: str = Field(default="hybrid", description="'dense', 'sparse', or 'hybrid'")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for BM25 in fusion")
```

**Read Output:**
```python
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
```

---

## 5. Code Pattern Requirements

### Naming Conventions

- **Classes**: PascalCase (`MemoryClusterStore`, `MemoryCluster`)
- **Functions**: snake_case (`write`, `read`, `consolidate_cluster`)
- **Variables**: snake_case (`cluster_index`, `prototype_embeddings`)
- **Constants**: SCREAMING_SNAKE_CASE (`DEFAULT_SIMILARITY_THRESHOLD`)

### Type Safety Requirements

- **Type hint coverage**: 100% for public API
- **Union syntax**: Use `|` operator
- **Generics**: Use builtin generics
- **Required import**: `from __future__ import annotations`

### Testing Approach

- **Framework**: pytest + pytest-asyncio
- **Coverage requirement**: ≥90%
- **Test patterns**: AAA (Arrange-Act-Assert)

**Required Test Cases:**
- `test_write_creates_new_cluster`
- `test_write_joins_existing_cluster`
- `test_consolidation_on_threshold`
- `test_read_returns_top_k`
- `test_temporal_decay_applied`
- `test_representative_selection`
- `test_index_rebuild`
- `test_concurrent_writes`

### Error Handling

- **Strategy**: Explicit raises
- **Custom exceptions**: `MCSError`, `ClusterNotFoundError`, `ConsolidationError`
- **Validation**: Input validation via Pydantic

### Architecture Patterns

- **Module structure**:
  - `src/prime/memory/mcs.py` - Main MCS class
  - `src/prime/memory/cluster.py` - MemoryCluster data class
  - `src/prime/memory/index.py` - FAISS index wrapper
- **Protocol**: Y-Encoder injected via `Encoder` protocol

---

## 6. Acceptance Criteria

### Definition of Done

- [ ] All functional requirements implemented
- [ ] All non-functional requirements met
- [ ] Unit tests passing with ≥90% coverage
- [ ] Integration test with FAISS index passing
- [ ] Consolidation benchmark meets <100ms target
- [ ] Type checking passes (`mypy --strict`)
- [ ] Documentation complete

### Validation Approach

1. **Unit Testing**: pytest with mock Y-Encoder
2. **Integration Testing**: MCS + FAISS + real Y-Encoder
3. **Benchmark Testing**: Write/read latency profiling
4. **Quality Testing**: Cluster purity (silhouette score) on test dataset

---

## 7. Dependencies

### Technical Assumptions

- Y-Encoder produces 1024-dimensional L2-normalized embeddings
- FAISS IndexFlatIP for cosine similarity search
- SQLite/PostgreSQL for metadata persistence (optional)

### External Integrations

| Integration | Type | Purpose |
|-------------|------|---------|
| Y-Encoder | Required | Content embedding |
| FAISS | Required | Vector similarity search |
| NumPy | Required | Vector operations |
| SQLAlchemy | Optional | Metadata persistence |

### Related Components

| Component | Relationship |
|-----------|--------------|
| Y-Encoder | Upstream dependency (encoding) |
| Predictor | Consumer (search with predicted embedding) |
| SSM | Trigger source (RETRIEVE_CONSOLIDATE) |
| PRIME | Parent orchestrator |

---

## 8. Configuration Schema

```python
class MCSConfig(BaseModel):
    """Configuration for Memory Cluster Store."""

    embedding_dim: int = Field(
        default=1024,
        description="Dimension of stored embeddings"
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cluster membership"
    )
    consolidation_threshold: int = Field(
        default=5,
        ge=2,
        description="Min cluster size to trigger consolidation"
    )
    max_clusters: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of clusters"
    )
    decay_factor: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Temporal decay factor per time unit"
    )
    decay_unit_seconds: int = Field(
        default=3600,
        description="Time unit for decay calculation (seconds)"
    )
    y_encoder_model: str = Field(
        default="google/gemma-embedding-300m",
        description="Model name for Y-Encoder"
    )
    index_type: str = Field(
        default="qdrant",
        description="Vector index type: 'faiss' or 'qdrant'"
    )

    # Hybrid Search Configuration
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable BM25 + vector hybrid search"
    )
    sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default weight for sparse (BM25) in RRF fusion"
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion method: 'rrf' (Reciprocal Rank Fusion) or 'rsf' (Relative Score Fusion)"
    )

    model_config = {"frozen": True}
```

---

## 9. Hybrid Search Implementation

### Qdrant Collection Setup

```python
from qdrant_client import QdrantClient, models

def create_hybrid_collection(client: QdrantClient, collection_name: str) -> None:
    """Create Qdrant collection with hybrid search support."""
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
```

### Hybrid Search Query

```python
from qdrant_client.models import SparseVector, NamedSparseVector

def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    dense_embedding: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    k: int = 5,
    sparse_weight: float = 0.3,
) -> list[ScoredPoint]:
    """Perform hybrid search with RRF fusion."""
    return client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_embedding,
                using="dense",
                limit=k * 2,
            ),
            models.Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="bm25",
                limit=k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    )
```

---

## 10. Data Structures

### MemoryCluster

```python
@dataclass
class MemoryCluster:
    """A cluster of semantically similar memories."""

    cluster_id: int
    prototype: np.ndarray  # L2-normalized centroid
    member_embeddings: list[np.ndarray]
    member_contents: list[str]
    member_metadata: list[dict[str, Any]]
    member_ids: list[str]
    is_consolidated: bool = False
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0

    @property
    def size(self) -> int:
        return len(self.member_contents)

    @property
    def representative_content(self) -> str:
        """Return content closest to prototype."""
        ...

    def add_member(
        self,
        embedding: np.ndarray,
        content: str,
        metadata: dict[str, Any],
        memory_id: str,
    ) -> None:
        """Add member and update prototype."""
        ...

    def consolidate(self) -> None:
        """Consolidate cluster into prototype."""
        ...
```

---

## Appendix: Source Traceability

| Requirement | Source Document | Section |
|-------------|-----------------|---------|
| FR-MCS-001 | PRIME-Project-Overview.md | 4.3.2 Architecture |
| FR-MCS-005 | PRIME-Project-Overview.md | 4.3.5 Consolidation |
| Compression Target | PRIME-Project-Overview.md | 1.3 Expected Outcomes |
| Hybrid Search | enhancement.md | Hybrid Search (BM25 + Vector) |
| Performance | strategic-intel.md | Technical Best Practices |
