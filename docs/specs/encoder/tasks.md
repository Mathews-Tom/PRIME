# ENC-001: Y-Encoder Task Breakdown

**Epic:** ENC-001 - Implement Y-Encoder
**Generated:** 2026-01-08
**Total Story Points:** 26 (with 20% buffer: 31)

---

## Task Summary

| Story ID | Title | Points | Dependencies | Critical Path |
|----------|-------|--------|--------------|---------------|
| ENC-001-S1 | Foundation: Exceptions, Config, Protocol | 3 | None | ✓ |
| ENC-001-S2 | Core: Pooling and YEncoder Implementation | 8 | ENC-001-S1 | ✓ |
| ENC-001-S3 | Testing: Comprehensive Test Suite | 5 | ENC-001-S2 | ✓ |
| ENC-001-S4 | Multi-Backend Support and Validation | 5 | ENC-001-S2 | |
| ENC-001-S5 | Enhancement: Caching and Performance | 5 | ENC-001-S3 | |

---

## Critical Path

```
ENC-001-S1 (3) → ENC-001-S2 (8) → ENC-001-S3 (5)
                       │
                       └──→ ENC-001-S4 (5)
                                │
                                └──→ ENC-001-S5 (5)
```

**Critical Path Duration:** 16 story points (S1 → S2 → S3)
**Full Epic Duration:** 26 story points (all stories)

---

## Story Details

### ENC-001-S1: Foundation - Exceptions, Config, Protocol

**Points:** 3
**Priority:** P0
**Dependencies:** None

#### Description
Establish the foundational types, configuration schema, and protocol definition for the Y-Encoder module. This creates the contract that YEncoder will implement.

#### Acceptance Criteria
- [ ] `src/prime/encoder/` directory structure created
- [ ] `exceptions.py` with EncoderError, ModelLoadError, EncodingError
- [ ] `config.py` with YEncoderConfig (frozen Pydantic model)
- [ ] `protocols.py` with Encoder Protocol definition
- [ ] All types pass mypy --strict
- [ ] Unit tests for config validation

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| ENC-001-S1-T1 | Create directory structure and __init__.py | XS |
| ENC-001-S1-T2 | Implement exceptions.py with error hierarchy | S |
| ENC-001-S1-T3 | Implement config.py with YEncoderConfig | S |
| ENC-001-S1-T4 | Implement protocols.py with Encoder Protocol | S |
| ENC-001-S1-T5 | Add unit tests for config validation | S |

#### Target Files
- `src/prime/encoder/__init__.py`
- `src/prime/encoder/exceptions.py`
- `src/prime/encoder/config.py`
- `src/prime/encoder/protocols.py`
- `tests/unit/encoder/test_config.py`

---

### ENC-001-S2: Core - Pooling and YEncoder Implementation

**Points:** 8
**Priority:** P0
**Dependencies:** ENC-001-S1

#### Description
Implement the core YEncoder class with model loading, tokenization, pooling strategies, and L2 normalization. This is the primary deliverable of the epic.

#### Acceptance Criteria
- [ ] `pooling.py` with mean/cls/max pooling strategies
- [ ] `y_encoder.py` with full YEncoder class
- [ ] Model loading from HuggingFace
- [ ] Single text encoding with encode()
- [ ] Batch encoding with encode_batch()
- [ ] L2 normalization producing unit vectors
- [ ] Device resolution (cuda/mps/cpu)
- [ ] Model info exposure via get_model_info()

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| ENC-001-S2-T1 | Implement pooling.py with mean/cls/max modes | M |
| ENC-001-S2-T2 | Implement YEncoder model loading | M |
| ENC-001-S2-T3 | Implement encode() single text method | M |
| ENC-001-S2-T4 | Implement encode_batch() method | M |
| ENC-001-S2-T5 | Implement L2 normalization | S |
| ENC-001-S2-T6 | Implement device resolution logic | S |
| ENC-001-S2-T7 | Implement get_model_info() | S |
| ENC-001-S2-T8 | Update __init__.py with exports | XS |

#### Target Files
- `src/prime/encoder/pooling.py`
- `src/prime/encoder/y_encoder.py`
- `src/prime/encoder/__init__.py`

---

### ENC-001-S3: Testing - Comprehensive Test Suite

**Points:** 5
**Priority:** P0
**Dependencies:** ENC-001-S2

#### Description
Create comprehensive test suite achieving ≥90% coverage. Tests must validate protocol compliance, core functionality, error handling, and semantic quality.

#### Acceptance Criteria
- [ ] Protocol compliance test
- [ ] Dimension correctness tests
- [ ] L2 normalization verification
- [ ] Batch vs single encoding consistency
- [ ] Truncation handling tests
- [ ] Error handling tests (empty input, invalid model)
- [ ] Pooling mode parameterized tests
- [ ] Semantic similarity quality test
- [ ] Test coverage ≥90%

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| ENC-001-S3-T1 | Create test fixtures (encoder_config, encoder) | S |
| ENC-001-S3-T2 | Implement protocol compliance tests | S |
| ENC-001-S3-T3 | Implement core functionality tests | M |
| ENC-001-S3-T4 | Implement error handling tests | S |
| ENC-001-S3-T5 | Implement pooling mode parameterized tests | S |
| ENC-001-S3-T6 | Implement semantic quality test | S |
| ENC-001-S3-T7 | Verify ≥90% coverage | S |

#### Target Files
- `tests/unit/encoder/test_encoder.py`
- `tests/unit/encoder/test_pooling.py`
- `tests/unit/encoder/conftest.py`

---

### ENC-001-S4: Multi-Backend Support and Validation

**Points:** 5
**Priority:** P1
**Dependencies:** ENC-001-S2

#### Description
Validate all supported encoder backends work correctly and document model trade-offs. Ensure dimension consistency across models.

#### Acceptance Criteria
- [ ] Gemma Embedding model works (1024-dim)
- [ ] MiniLM model works (384-dim)
- [ ] BGE Large model works (1024-dim)
- [ ] Qwen3 0.6B model works (1024-dim)
- [ ] Model switching at runtime validated
- [ ] Dimension validation at load time
- [ ] Documentation of model trade-offs

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| ENC-001-S4-T1 | Test Gemma Embedding backend | M |
| ENC-001-S4-T2 | Test MiniLM backend | S |
| ENC-001-S4-T3 | Test BGE Large backend | M |
| ENC-001-S4-T4 | Test Qwen3 backend | M |
| ENC-001-S4-T5 | Add dimension validation at load | S |
| ENC-001-S4-T6 | Document model trade-offs | S |

#### Target Files
- `tests/integration/encoder/test_backends.py`
- `docs/specs/encoder/models.md`

---

### ENC-001-S5: Enhancement - Caching and Performance

**Points:** 5
**Priority:** P2
**Dependencies:** ENC-001-S3

#### Description
Implement optional LRU caching for repeated content and optimize performance. Validate latency targets are met.

#### Acceptance Criteria
- [ ] LRU cache implementation with configurable size
- [ ] Cache key computation (SHA256)
- [ ] Cache hit/miss tracking
- [ ] Single encoding latency <50ms p50
- [ ] Batch encoding (32) latency <500ms
- [ ] Performance benchmark tests

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| ENC-001-S5-T1 | Implement LRU cache setup | M |
| ENC-001-S5-T2 | Implement cache key computation | S |
| ENC-001-S5-T3 | Add cache integration to encode() | M |
| ENC-001-S5-T4 | Create latency benchmark tests | M |
| ENC-001-S5-T5 | Validate latency targets | S |

#### Target Files
- `src/prime/encoder/y_encoder.py` (update)
- `tests/benchmark/encoder/test_latency.py`

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
- MCS-001: Requires Y-Encoder for content embedding
- PRED-001: Requires Y-Encoder for training targets

**Blocked By:**
- None (foundational component)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model download failures | Medium | High | Cache models locally, test offline |
| GPU memory exhaustion | Medium | Medium | CPU fallback, batch size config |
| Embedding dim mismatch | Low | High | Validate at load time |
| Latency targets missed | Medium | Medium | Smaller model fallback |

---

## Definition of Done (Epic Level)

- [ ] All P0 stories completed
- [ ] Test coverage ≥90%
- [ ] mypy --strict passes
- [ ] ruff check passes
- [ ] All acceptance criteria met
- [ ] Documentation complete
- [ ] Ready for MCS-001 and PRED-001 integration
