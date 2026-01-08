# API-001: PRIME API Task Breakdown

**Epic:** API-001 - Implement PRIME API
**Generated:** 2026-01-08
**Total Story Points:** 42 (with 20% buffer: 50)

---

## Task Summary

| Story ID | Title | Points | Dependencies | Critical Path |
|----------|-------|--------|--------------|---------------|
| API-001-S1 | Foundation: Types, Config, Exceptions | 3 | ENC-001, SSM-001, MCS-001, PRED-001 | ✓ |
| API-001-S2 | Core: PRIME Orchestration Class | 8 | API-001-S1 | ✓ |
| API-001-S3 | API: FastAPI Application and Routes | 8 | API-001-S2 | ✓ |
| API-001-S4 | Middleware: Authentication and Rate Limiting | 5 | API-001-S3 | ✓ |
| API-001-S5 | Evaluation: RAGAS Integration | 5 | API-001-S3 | |
| API-001-S6 | Adapters: LangChain and LlamaIndex | 5 | API-001-S2 | |
| API-001-S7 | Client: Python SDK | 3 | API-001-S3 | |
| API-001-S8 | Testing: Comprehensive Test Suite | 5 | API-001-S4 | ✓ |

---

## Critical Path

```
ENC-001, SSM-001, MCS-001, PRED-001 (external)
    │
    └──▶ API-001-S1 (3)
              │
              └──▶ API-001-S2 (8) ──┬──▶ API-001-S3 (8) ──▶ API-001-S4 (5) ──▶ API-001-S8 (5)
                                    │           │
                                    │           ├──▶ API-001-S5 (5)
                                    │           │
                                    │           └──▶ API-001-S7 (3)
                                    │
                                    └──▶ API-001-S6 (5)
```

**Critical Path Duration:** 29 story points (S1 → S2 → S3 → S4 → S8)
**Full Epic Duration:** 42 story points (all stories)

---

## Story Details

### API-001-S1: Foundation - Types, Config, Exceptions

**Points:** 3
**Priority:** P0
**Dependencies:** ENC-001, SSM-001, MCS-001, PRED-001

#### Description
Establish the foundational types, configuration schema, and exception hierarchy for the PRIME API module. This includes PRIMEConfig, APIConfig, RAGASConfig, and all shared types.

#### Acceptance Criteria
- [ ] `src/prime/` directory structure created
- [ ] `types.py` with ActionState, MemoryReadResult, MemoryWriteResult, PRIMEResponse, PRIMEDiagnostics
- [ ] `config.py` with PRIMEConfig, APIConfig, RAGASConfig (frozen Pydantic models)
- [ ] `exceptions.py` with PRIMEError hierarchy
- [ ] All types pass mypy --strict
- [ ] Unit tests for config validation

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S1-T1 | Create directory structure | XS |
| API-001-S1-T2 | Implement types.py with all shared types | M |
| API-001-S1-T3 | Implement config.py with all config classes | M |
| API-001-S1-T4 | Implement exceptions.py with error hierarchy | S |
| API-001-S1-T5 | Add unit tests for config validation | S |

#### Target Files
- `src/prime/__init__.py`
- `src/prime/types.py`
- `src/prime/config.py`
- `src/prime/exceptions.py`
- `tests/unit/api/test_config.py`

---

### API-001-S2: Core - PRIME Orchestration Class

**Points:** 8
**Priority:** P0
**Dependencies:** API-001-S1

#### Description
Implement the main PRIME orchestration class that integrates SSM, Predictor, MCS, and Y-Encoder into a unified system with process_turn(), record_response(), write_external_knowledge(), search_memory(), and get_diagnostics() methods.

#### Acceptance Criteria
- [ ] PRIME class with component initialization
- [ ] process_turn() orchestrates full pipeline
- [ ] SSM integration for boundary detection
- [ ] Predictor integration for target embedding
- [ ] MCS integration for memory search/write
- [ ] record_response() stores LLM responses
- [ ] write_external_knowledge() for document ingestion
- [ ] search_memory() for direct search
- [ ] get_diagnostics() for system health
- [ ] Structured logging with structlog

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S2-T1 | Implement PRIME __init__ with component initialization | M |
| API-001-S2-T2 | Implement process_turn() method | L |
| API-001-S2-T3 | Implement record_response() method | M |
| API-001-S2-T4 | Implement write_external_knowledge() method | S |
| API-001-S2-T5 | Implement search_memory() method | S |
| API-001-S2-T6 | Implement get_diagnostics() method | M |
| API-001-S2-T7 | Add structured logging | S |
| API-001-S2-T8 | Unit tests for PRIME class | M |

#### Target Files
- `src/prime/prime.py`
- `src/prime/__init__.py` (update exports)
- `tests/unit/api/test_prime.py`

---

### API-001-S3: API - FastAPI Application and Routes

**Points:** 8
**Priority:** P0
**Dependencies:** API-001-S2

#### Description
Implement the FastAPI application factory and REST endpoints: /process, /memory/write, /memory/search, /diagnostics, /health, /clusters, and /config.

#### Acceptance Criteria
- [ ] FastAPI app factory with lifespan handler
- [ ] POST /process endpoint
- [ ] POST /memory/write endpoint
- [ ] POST /memory/search endpoint
- [ ] GET /diagnostics endpoint
- [ ] GET /health endpoint
- [ ] GET /clusters endpoint
- [ ] PUT /config endpoint
- [ ] Request validation with Pydantic
- [ ] Structured error responses
- [ ] OpenAPI documentation at /docs

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S3-T1 | Implement app.py factory with lifespan | M |
| API-001-S3-T2 | Implement dependencies.py for DI | S |
| API-001-S3-T3 | Implement process route | M |
| API-001-S3-T4 | Implement memory routes (write, search) | M |
| API-001-S3-T5 | Implement diagnostics routes (diagnostics, health) | S |
| API-001-S3-T6 | Implement clusters routes | S |
| API-001-S3-T7 | Implement config route | S |
| API-001-S3-T8 | Add exception handlers | S |
| API-001-S3-T9 | Add request timing middleware | S |

#### Target Files
- `src/prime/api/__init__.py`
- `src/prime/api/app.py`
- `src/prime/api/dependencies.py`
- `src/prime/api/routes/__init__.py`
- `src/prime/api/routes/process.py`
- `src/prime/api/routes/memory.py`
- `src/prime/api/routes/diagnostics.py`
- `src/prime/api/routes/clusters.py`
- `src/prime/api/routes/config.py`

---

### API-001-S4: Middleware - Authentication and Rate Limiting

**Points:** 5
**Priority:** P0
**Dependencies:** API-001-S3

#### Description
Implement API key authentication middleware and sliding window rate limiting middleware with proper headers and error responses.

#### Acceptance Criteria
- [ ] APIKeyMiddleware for X-API-Key header validation
- [ ] Public paths bypass (health, docs)
- [ ] RateLimitMiddleware with sliding window
- [ ] Rate limit headers (X-RateLimit-*)
- [ ] 429 response with Retry-After header
- [ ] Request logging middleware
- [ ] CORS middleware configuration

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S4-T1 | Implement APIKeyMiddleware | M |
| API-001-S4-T2 | Implement RateLimitMiddleware | M |
| API-001-S4-T3 | Implement logging middleware | S |
| API-001-S4-T4 | Configure CORS middleware | S |
| API-001-S4-T5 | Middleware unit tests | M |

#### Target Files
- `src/prime/api/middleware/__init__.py`
- `src/prime/api/middleware/auth.py`
- `src/prime/api/middleware/rate_limit.py`
- `src/prime/api/middleware/logging.py`
- `tests/unit/api/test_middleware.py`

---

### API-001-S5: Evaluation - RAGAS Integration

**Points:** 5
**Priority:** P1
**Dependencies:** API-001-S3

#### Description
Implement RAGAS evaluation integration for reference-free RAG quality metrics: Faithfulness, Context Precision, Answer Relevancy, and optional Context Recall.

#### Acceptance Criteria
- [ ] RAGASEvaluator class with LLM setup
- [ ] evaluate() for single sample evaluation
- [ ] evaluate_batch() for batch evaluation
- [ ] Faithfulness metric
- [ ] Context Precision metric
- [ ] Answer Relevancy metric
- [ ] Context Recall (optional, requires ground truth)
- [ ] POST /diagnostics/eval endpoint
- [ ] Tests for RAGAS evaluation

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S5-T1 | Implement RAGASEvaluator class | M |
| API-001-S5-T2 | Implement evaluate() method | M |
| API-001-S5-T3 | Implement evaluate_batch() method | S |
| API-001-S5-T4 | Add /diagnostics/eval route | S |
| API-001-S5-T5 | RAGAS integration tests | M |

#### Target Files
- `src/prime/evaluation/__init__.py`
- `src/prime/evaluation/ragas.py`
- `src/prime/api/routes/diagnostics.py` (update)
- `tests/integration/test_ragas.py`

---

### API-001-S6: Adapters - LangChain and LlamaIndex

**Points:** 5
**Priority:** P2
**Dependencies:** API-001-S2

#### Description
Implement LangChain BaseRetriever adapter and LlamaIndex retriever adapter for framework ecosystem compatibility.

#### Acceptance Criteria
- [ ] PRIMERetriever implements LangChain BaseRetriever
- [ ] _get_relevant_documents() method
- [ ] Support for both process_turn and direct search modes
- [ ] LlamaIndex retriever implementation
- [ ] Adapter tests

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S6-T1 | Implement PRIMERetriever (LangChain) | M |
| API-001-S6-T2 | Implement LlamaIndex retriever | M |
| API-001-S6-T3 | Adapter integration tests | M |
| API-001-S6-T4 | Usage examples | S |

#### Target Files
- `src/prime/adapters/__init__.py`
- `src/prime/adapters/langchain.py`
- `src/prime/adapters/llamaindex.py`
- `tests/integration/test_adapters.py`

---

### API-001-S7: Client - Python SDK

**Points:** 3
**Priority:** P1
**Dependencies:** API-001-S3

#### Description
Implement Python client SDK for PRIME API with process_turn(), search(), write_memory(), and health() methods.

#### Acceptance Criteria
- [ ] PRIMEClient class with httpx
- [ ] process_turn() method
- [ ] search() method
- [ ] write_memory() method
- [ ] health() method
- [ ] Context manager support
- [ ] Client SDK tests

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S7-T1 | Implement PRIMEClient class | M |
| API-001-S7-T2 | Implement all API methods | M |
| API-001-S7-T3 | Client SDK tests | S |

#### Target Files
- `src/prime/client/__init__.py`
- `src/prime/client/client.py`
- `tests/integration/test_client.py`

---

### API-001-S8: Testing - Comprehensive Test Suite

**Points:** 5
**Priority:** P0
**Dependencies:** API-001-S4

#### Description
Create comprehensive test suite achieving ≥85% coverage including unit tests, integration tests, and performance tests.

#### Acceptance Criteria
- [ ] PRIME class unit tests
- [ ] Route unit tests
- [ ] Middleware unit tests
- [ ] API integration tests with httpx.AsyncClient
- [ ] Authentication tests
- [ ] Rate limiting tests
- [ ] Performance benchmarks
- [ ] Test coverage ≥85%

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| API-001-S8-T1 | Create test fixtures (conftest.py) | S |
| API-001-S8-T2 | PRIME class unit tests | M |
| API-001-S8-T3 | Route integration tests | M |
| API-001-S8-T4 | Auth and rate limit tests | M |
| API-001-S8-T5 | Performance benchmarks | M |
| API-001-S8-T6 | Verify ≥85% coverage | S |

#### Target Files
- `tests/unit/api/test_prime.py`
- `tests/unit/api/test_routes.py`
- `tests/unit/api/test_middleware.py`
- `tests/integration/test_api.py`
- `tests/fixtures/conftest.py`
- `tests/benchmark/api/test_latency.py`

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
None (final integration layer)

**Blocked By:**
- ENC-001: Y-Encoder for content embedding
- SSM-001: Semantic State Monitor for boundary detection
- MCS-001: Memory Cluster Store for memory operations
- PRED-001: Embedding Predictor for target prediction

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Component integration failures | Medium | High | Mock-based unit tests, integration tests |
| Latency exceeds 200ms target | Medium | High | Performance profiling, caching |
| Rate limiter memory growth | Medium | Medium | Time-based cleanup, Redis backend |
| RAGAS LLM costs | Medium | Medium | Batch evaluation, sampling |
| Auth middleware bypass | Low | High | Security testing, code review |

---

## Key Flows

### Process Turn Flow
```
1. Validate input (Pydantic)
2. Authenticate (X-API-Key)
3. Rate limit check
4. PRIME.process_turn():
   a. Y-Encoder.encode(input)
   b. SSM.update(embedding)
   c. If boundary crossed or force_retrieval:
      i. Get context window
      ii. Predictor.predict(context, current)
      iii. MCS.search(predicted_embedding)
   d. If consolidate: MCS.consolidate()
5. Format response
6. Return with timing headers
```

### Memory Write Flow
```
1. Validate content
2. Y-Encoder.encode(content)
3. MCS.write(content, embedding, metadata)
4. Return memory_id, cluster_id
```

---

## Performance Targets

| Operation | Target p50 | Target p99 |
|-----------|------------|------------|
| process_turn | <200ms | <500ms |
| memory/write | <50ms | <100ms |
| memory/search | <80ms | <150ms |
| health | <10ms | <20ms |

---

## Definition of Done (Epic Level)

- [ ] All P0 stories completed
- [ ] Test coverage ≥85%
- [ ] mypy --strict passes
- [ ] ruff check passes
- [ ] End-to-end latency <200ms p50
- [ ] API throughput >100 QPS
- [ ] Authentication working
- [ ] Rate limiting working
- [ ] OpenAPI documentation at /docs
- [ ] Docker containerization complete
- [ ] Ready for production deployment
