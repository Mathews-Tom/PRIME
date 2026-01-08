# SSM-001: Semantic State Monitor Task Breakdown

**Epic:** SSM-001 - Implement Semantic State Monitor
**Generated:** 2026-01-08
**Total Story Points:** 21 (with 20% buffer: 25)

---

## Task Summary

| Story ID | Title | Points | Dependencies | Critical Path |
|----------|-------|--------|--------------|---------------|
| SSM-001-S1 | Foundation: Types, Config, Exceptions | 3 | None | ✓ |
| SSM-001-S2 | Core: SSM Implementation | 8 | SSM-001-S1 | ✓ |
| SSM-001-S3 | Testing: Comprehensive Test Suite | 5 | SSM-001-S2 | ✓ |
| SSM-001-S4 | Integration: X-Encoder and Observability | 5 | SSM-001-S3 | |

---

## Critical Path

```
SSM-001-S1 (3) → SSM-001-S2 (8) → SSM-001-S3 (5)
                       │
                       └──→ SSM-001-S4 (5)
```

**Critical Path Duration:** 16 story points (S1 → S2 → S3)
**Full Epic Duration:** 21 story points (all stories)

---

## Story Details

### SSM-001-S1: Foundation - Types, Config, Exceptions

**Points:** 3
**Priority:** P0
**Dependencies:** None

#### Description
Establish the foundational types, configuration schema, and exception hierarchy for the SSM module. This creates the ActionState enum, SemanticStateUpdate result type, and SSMConfig.

#### Acceptance Criteria
- [ ] `src/prime/ssm/` directory structure created
- [ ] `ssm_types.py` with ActionState enum and SemanticStateUpdate
- [ ] `ssm_config.py` with SSMConfig (frozen Pydantic model)
- [ ] `exceptions.py` with SSMError hierarchy
- [ ] All types pass mypy --strict
- [ ] Unit tests for config validation

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| SSM-001-S1-T1 | Create directory structure and __init__.py | XS |
| SSM-001-S1-T2 | Implement ActionState enum (CONTINUE, PREPARE, RETRIEVE, RETRIEVE_CONSOLIDATE) | S |
| SSM-001-S1-T3 | Implement SemanticStateUpdate Pydantic model | S |
| SSM-001-S1-T4 | Implement SSMConfig with thresholds | S |
| SSM-001-S1-T5 | Implement SSMError exception hierarchy | S |
| SSM-001-S1-T6 | Add unit tests for config validation | S |

#### Target Files
- `src/prime/ssm/__init__.py`
- `src/prime/ssm/ssm_types.py`
- `src/prime/ssm/ssm_config.py`
- `src/prime/ssm/exceptions.py`
- `tests/unit/ssm/test_config.py`

---

### SSM-001-S2: Core - SSM Implementation

**Points:** 8
**Priority:** P0
**Dependencies:** SSM-001-S1

#### Description
Implement the core SemanticStateMonitor class with sliding window buffer, Ward variance calculation, EMA smoothing, and action state determination.

#### Acceptance Criteria
- [ ] SemanticStateMonitor class with `update()` method
- [ ] Sliding window buffer using deque (FIFO)
- [ ] Ward variance calculation: `Var(||e_i - centroid||)`
- [ ] EMA smoothing: `smoothed = α * current + (1-α) * previous`
- [ ] Action state determination based on thresholds
- [ ] Encoder protocol injection for X-Encoder
- [ ] reset() method for new conversations
- [ ] get_state() for observability

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| SSM-001-S2-T1 | Implement sliding window buffer with deque | S |
| SSM-001-S2-T2 | Implement centroid calculation | S |
| SSM-001-S2-T3 | Implement Ward variance calculation | M |
| SSM-001-S2-T4 | Implement EMA smoothing | S |
| SSM-001-S2-T5 | Implement action state determination | M |
| SSM-001-S2-T6 | Implement update() method (main entry point) | M |
| SSM-001-S2-T7 | Implement reset() method | S |
| SSM-001-S2-T8 | Implement get_state() for observability | S |
| SSM-001-S2-T9 | Update __init__.py with exports | XS |

#### Target Files
- `src/prime/ssm/ssm.py`
- `src/prime/ssm/__init__.py`

---

### SSM-001-S3: Testing - Comprehensive Test Suite

**Points:** 5
**Priority:** P0
**Dependencies:** SSM-001-S2

#### Description
Create comprehensive test suite achieving ≥90% coverage. Tests must validate boundary detection, EMA smoothing, action state transitions, and error handling using mock encoder.

#### Acceptance Criteria
- [ ] MockEncoder for isolated testing
- [ ] Boundary detection tests (topic change detected)
- [ ] No false positives on same topic
- [ ] EMA smoothing reduces noise
- [ ] Action state transition tests
- [ ] Window buffer FIFO behavior
- [ ] Error handling tests (empty input, dimension mismatch)
- [ ] Reset clears state
- [ ] Test coverage ≥90%

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| SSM-001-S3-T1 | Create MockEncoder for testing | S |
| SSM-001-S3-T2 | Create test fixtures (mock_encoder, ssm_config, ssm) | S |
| SSM-001-S3-T3 | Implement boundary detection tests | M |
| SSM-001-S3-T4 | Implement EMA smoothing tests | S |
| SSM-001-S3-T5 | Implement action state tests | S |
| SSM-001-S3-T6 | Implement window buffer tests | S |
| SSM-001-S3-T7 | Implement error handling tests | S |
| SSM-001-S3-T8 | Verify ≥90% coverage | S |

#### Target Files
- `tests/unit/ssm/test_ssm.py`
- `tests/unit/ssm/conftest.py`

---

### SSM-001-S4: Integration - X-Encoder and Observability

**Points:** 5
**Priority:** P1
**Dependencies:** SSM-001-S3

#### Description
Integrate SSM with real X-Encoder (using YEncoder with Encoder protocol), add performance benchmarks, and validate latency targets.

#### Acceptance Criteria
- [ ] Integration test with YEncoder (using MiniLM for speed)
- [ ] Dimension validation at runtime
- [ ] Variance calculation latency <5ms p50
- [ ] End-to-end latency <50ms (encoding + SSM)
- [ ] Observability metrics (turn_number, window_size, variance)
- [ ] Documentation of threshold tuning

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| SSM-001-S4-T1 | Create integration test with YEncoder | M |
| SSM-001-S4-T2 | Add dimension validation | S |
| SSM-001-S4-T3 | Create latency benchmark tests | M |
| SSM-001-S4-T4 | Validate performance targets | S |
| SSM-001-S4-T5 | Document threshold tuning guide | S |

#### Target Files
- `tests/integration/ssm/test_ssm_encoder.py`
- `tests/benchmark/ssm/test_latency.py`
- `docs/specs/ssm/tuning.md`

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
- API-001: Uses SSM for retrieval triggering

**Blocked By:**
- None (uses Encoder Protocol, not direct ENC-001 dependency)

**Protocol Dependency:**
- Uses Encoder Protocol (defined in ENC-001) for X-Encoder
- Can use any encoder implementing the protocol

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Variance threshold tuning | Medium | Medium | Configurable, provide tuning guide |
| False positives (unnecessary retrieval) | Medium | Medium | EMA smoothing reduces spikes |
| False negatives (missed boundaries) | Medium | High | High recall target, force_retrieval override |
| X-Encoder latency dominating | High | Medium | SSM calc is O(n), encoder is bottleneck |

---

## Key Algorithms

### Ward Variance Calculation
```
W = {e₁, e₂, ..., eₙ}
μ = mean(W)
dᵢ = ||eᵢ - μ||₂
Variance = Var({d₁, d₂, ..., dₙ})
```

### EMA Smoothing
```
sₜ = α * vₜ + (1-α) * sₜ₋₁
where α = 0.3 (default)
```

### Action State Thresholds
```
CONTINUE:           v < 0.5θ
PREPARE:            0.5θ ≤ v < θ
RETRIEVE:           θ ≤ v < 2θ
RETRIEVE_CONSOLIDATE: v ≥ 2θ
```

---

## Definition of Done (Epic Level)

- [ ] All P0 stories completed
- [ ] Test coverage ≥90%
- [ ] mypy --strict passes
- [ ] ruff check passes
- [ ] Retrieval reduction validated (60-70% vs baseline)
- [ ] Boundary detection precision >85%
- [ ] Boundary detection recall >90%
- [ ] Variance calculation latency <5ms p50
- [ ] Ready for API-001 integration
