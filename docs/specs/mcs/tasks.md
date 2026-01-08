# MCS-001: Memory Cluster Store Task Breakdown

**Epic:** MCS-001 - Implement Memory Cluster Store
**Generated:** 2026-01-08
**Total Story Points:** 34 (with 20% buffer: 41)

---

## Task Summary

| Story ID | Title | Points | Dependencies | Critical Path |
|----------|-------|--------|--------------|---------------|
| MCS-001-S1 | Foundation: Types, Config, Exceptions | 3 | ENC-001 | ✓ |
| MCS-001-S2 | Core: MemoryCluster and Cluster Management | 5 | MCS-001-S1 | ✓ |
| MCS-001-S3 | Vector Index: Qdrant Integration | 8 | MCS-001-S1 | ✓ |
| MCS-001-S4 | Core: MCS Write and Read Paths | 8 | MCS-001-S2, MCS-001-S3 | ✓ |
| MCS-001-S5 | Hybrid Search: BM25 and RRF Fusion | 5 | MCS-001-S4 | |
| MCS-001-S6 | Testing: Comprehensive Test Suite | 5 | MCS-001-S4 | ✓ |

---

## Critical Path

```
ENC-001 (external)
    │
    └──▶ MCS-001-S1 (3)
              │
              ├──▶ MCS-001-S2 (5) ──┐
              │                     │
              └──▶ MCS-001-S3 (8) ──┴──▶ MCS-001-S4 (8) ──▶ MCS-001-S6 (5)
                                              │
                                              └──▶ MCS-001-S5 (5)
```

**Critical Path Duration:** 24 story points (S1 → S3 → S4 → S6)
**Full Epic Duration:** 34 story points (all stories)

---

## Story Details

### MCS-001-S1: Foundation - Types, Config, Exceptions

**Points:** 3
**Priority:** P0
**Dependencies:** ENC-001 (for Encoder protocol)

#### Description
Establish the foundational types, configuration schema, and exception hierarchy for the MCS module. This includes input/output models for write/read operations and MCSConfig.

#### Acceptance Criteria
- [ ] `src/prime/mcs/` directory structure created
- [ ] `types.py` with MemoryWriteInput, MemoryWriteResult, MemoryReadInput, MemoryReadResult
- [ ] `mcs_config.py` with MCSConfig (frozen Pydantic model)
- [ ] `exceptions.py` with MCSError, ClusterNotFoundError, ConsolidationError
- [ ] All types pass mypy --strict
- [ ] Unit tests for config validation

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| MCS-001-S1-T1 | Create directory structure and __init__.py | XS |
| MCS-001-S1-T2 | Implement types.py with I/O models | S |
| MCS-001-S1-T3 | Implement mcs_config.py with thresholds | S |
| MCS-001-S1-T4 | Implement exceptions.py | S |
| MCS-001-S1-T5 | Add unit tests for config validation | S |

#### Target Files
- `src/prime/mcs/__init__.py`
- `src/prime/mcs/types.py`
- `src/prime/mcs/mcs_config.py`
- `src/prime/mcs/exceptions.py`
- `tests/unit/mcs/test_config.py`

---

### MCS-001-S2: Core - MemoryCluster and Cluster Management

**Points:** 5
**Priority:** P0
**Dependencies:** MCS-001-S1

#### Description
Implement the MemoryCluster dataclass with member management, prototype computation, and consolidation logic.

#### Acceptance Criteria
- [ ] MemoryCluster dataclass with all required fields
- [ ] add_member() updates prototype as L2-normalized centroid
- [ ] representative_content property returns closest to prototype
- [ ] consolidate() marks cluster immutable, keeps only representative
- [ ] similarity_to() computes cosine similarity to prototype
- [ ] Unit tests for all cluster operations

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| MCS-001-S2-T1 | Implement MemoryCluster dataclass | M |
| MCS-001-S2-T2 | Implement add_member with prototype update | M |
| MCS-001-S2-T3 | Implement representative_content property | S |
| MCS-001-S2-T4 | Implement consolidate() method | M |
| MCS-001-S2-T5 | Implement similarity_to() method | S |
| MCS-001-S2-T6 | Unit tests for cluster operations | M |

#### Target Files
- `src/prime/mcs/cluster.py`
- `tests/unit/mcs/test_cluster.py`

---

### MCS-001-S3: Vector Index - Qdrant Integration

**Points:** 8
**Priority:** P0
**Dependencies:** MCS-001-S1

#### Description
Implement the Qdrant vector index wrapper with support for both dense and sparse vectors, enabling hybrid search capability.

#### Acceptance Criteria
- [ ] VectorIndex protocol definition
- [ ] QdrantIndex class with collection management
- [ ] add() supports both dense and sparse vectors
- [ ] search_dense() for vector-only search
- [ ] search_hybrid() with RRF fusion
- [ ] remove() and rebuild() methods
- [ ] Integration tests with Qdrant

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| MCS-001-S3-T1 | Implement index.py protocol | S |
| MCS-001-S3-T2 | Implement QdrantIndex initialization | M |
| MCS-001-S3-T3 | Implement _ensure_collection() | M |
| MCS-001-S3-T4 | Implement add() with dense/sparse vectors | M |
| MCS-001-S3-T5 | Implement search_dense() | M |
| MCS-001-S3-T6 | Implement search_hybrid() with RRF | L |
| MCS-001-S3-T7 | Implement remove() and rebuild() | S |
| MCS-001-S3-T8 | Integration tests with Qdrant | M |

#### Target Files
- `src/prime/mcs/index.py`
- `src/prime/mcs/qdrant_index.py`
- `tests/integration/mcs/test_qdrant.py`

---

### MCS-001-S4: Core - MCS Write and Read Paths

**Points:** 8
**Priority:** P0
**Dependencies:** MCS-001-S2, MCS-001-S3

#### Description
Implement the main MemoryClusterStore class with write (cluster assignment) and read (similarity search) paths, including consolidation triggering.

#### Acceptance Criteria
- [ ] MemoryClusterStore class with Y-Encoder injection
- [ ] write() encodes content, assigns to cluster
- [ ] write() creates new cluster if similarity < threshold
- [ ] write() triggers consolidation at threshold
- [ ] read() performs similarity search with temporal decay
- [ ] get_cluster() and get_stats() methods
- [ ] Unit tests with mock encoder

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| MCS-001-S4-T1 | Implement MCS __init__ with encoder/config | M |
| MCS-001-S4-T2 | Implement _find_nearest_cluster() | M |
| MCS-001-S4-T3 | Implement _create_cluster() | S |
| MCS-001-S4-T4 | Implement write() method | L |
| MCS-001-S4-T5 | Implement _consolidate_cluster() | M |
| MCS-001-S4-T6 | Implement read() with temporal decay | L |
| MCS-001-S4-T7 | Implement get_cluster() and get_stats() | S |
| MCS-001-S4-T8 | Update __init__.py with exports | XS |

#### Target Files
- `src/prime/mcs/mcs.py`
- `src/prime/mcs/__init__.py`

---

### MCS-001-S5: Hybrid Search - BM25 and RRF Fusion

**Points:** 5
**Priority:** P1
**Dependencies:** MCS-001-S4

#### Description
Implement BM25 sparse vector generation and integrate hybrid search (dense + sparse) with Reciprocal Rank Fusion into the read path.

#### Acceptance Criteria
- [ ] BM25Tokenizer for sparse vector generation
- [ ] tokenize() with lowercase and min_length filtering
- [ ] encode() returns (indices, values) for sparse vector
- [ ] write() stores sparse vectors alongside dense
- [ ] read() supports search_mode: dense/sparse/hybrid
- [ ] RRF fusion combines results from both indexes
- [ ] Performance benchmarks showing hybrid vs dense improvement

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| MCS-001-S5-T1 | Implement BM25Tokenizer | M |
| MCS-001-S5-T2 | Implement tokenize() method | S |
| MCS-001-S5-T3 | Implement encode() for sparse vectors | S |
| MCS-001-S5-T4 | Integrate sparse vectors into write() | M |
| MCS-001-S5-T5 | Implement search mode selection in read() | M |
| MCS-001-S5-T6 | Test hybrid search vs dense-only | M |

#### Target Files
- `src/prime/mcs/sparse.py`
- `src/prime/mcs/mcs.py` (update)
- `tests/unit/mcs/test_hybrid.py`

---

### MCS-001-S6: Testing - Comprehensive Test Suite

**Points:** 5
**Priority:** P0
**Dependencies:** MCS-001-S4

#### Description
Create comprehensive test suite achieving ≥90% coverage including write, read, consolidation, temporal decay, and cluster management tests.

#### Acceptance Criteria
- [ ] MockEncoder for isolated testing
- [ ] Write tests (new cluster, join existing, force new)
- [ ] Consolidation tests at threshold
- [ ] Read tests (top-k, empty store, min_similarity filter)
- [ ] Temporal decay tests
- [ ] Cluster unit tests (add_member, consolidate, representative)
- [ ] Statistics tests
- [ ] Test coverage ≥90%

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| MCS-001-S6-T1 | Create MockEncoder and fixtures | S |
| MCS-001-S6-T2 | Implement write tests | M |
| MCS-001-S6-T3 | Implement consolidation tests | S |
| MCS-001-S6-T4 | Implement read tests | M |
| MCS-001-S6-T5 | Implement temporal decay tests | S |
| MCS-001-S6-T6 | Implement cluster unit tests | M |
| MCS-001-S6-T7 | Verify ≥90% coverage | S |

#### Target Files
- `tests/unit/mcs/test_mcs.py`
- `tests/unit/mcs/test_cluster.py`
- `tests/unit/mcs/conftest.py`

---

## Effort Legend

| Size | Story Points | Time Estimate |
|------|--------------|---------------|
| XS | 0.5 | <1 hour |
| S | 1 | 1-2 hours |
| M | 2 | 2-4 hours |
| L | 3 | 4-8 hours |
| XL | 5 | 1-2 days |

---

## Dependencies on Other Epics

**Blocks:**
- API-001: Uses MCS for memory storage and retrieval

**Blocked By:**
- ENC-001: Requires Y-Encoder for content embedding

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Qdrant connection issues | Low | High | In-memory fallback, health checks |
| Index corruption | Low | High | Regular backups, rebuild capability |
| Memory pressure | Medium | Medium | Max cluster limit, consolidation |
| BM25 vocabulary explosion | Medium | Low | Vocabulary pruning |
| Write latency exceeds target | Medium | Medium | Async consolidation |

---

## Key Algorithms

### Cluster Assignment
```
1. Encode content with Y-Encoder
2. Search for nearest cluster prototype
3. If similarity > θ_cluster (0.85): join cluster
4. Else: create new cluster
5. Update prototype as L2-normalized centroid
6. If cluster.size >= τ_consolidate (5): consolidate
```

### Temporal Decay
```
score_adjusted = score × decay_factor^(age_units)
where age_units = (now - last_access) / decay_unit_seconds
```

### Reciprocal Rank Fusion (RRF)
```
RRF(d) = Σ 1/(k + rank_i(d))
where k = 60 (constant), rank_i(d) = rank of document d in result list i
```

---

## Definition of Done (Epic Level)

- [ ] All P0 stories completed
- [ ] Test coverage ≥90%
- [ ] mypy --strict passes
- [ ] ruff check passes
- [ ] Storage compression 3-5× achieved
- [ ] Write latency <20ms p50
- [ ] Read latency <30ms p50
- [ ] Cluster purity >0.85 (silhouette score)
- [ ] Ready for API-001 integration
