# Semantic State Monitor (SSM) Specification

## 1. Overview

### Purpose and Business Value

The Semantic State Monitor (SSM) is PRIME's intelligent retrieval trigger component. It monitors the semantic trajectory of conversations and triggers retrieval **only when a significant semantic boundary is crossed**, reducing unnecessary retrievals by 60-70%.

**Business Value:**
- Reduces compute costs by eliminating 60-70% of retrieval operations
- Improves response quality by avoiding context pollution from irrelevant retrievals
- Enables predictive caching through PREPARE state detection

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Retrieval Trigger Reduction | 60-70% | vs. every-turn baseline |
| Boundary Detection Precision | >85% | Manual evaluation on test set |
| Boundary Detection Recall | >90% | Topic change detection accuracy |
| Latency Overhead | <10ms | p95 variance calculation time |

### Target Users

- PRIME core system (internal API)
- Conversation orchestration layer
- Observability/debugging tools (MIMIR)

---

## 2. Functional Requirements

### Core Capabilities

- **FR-SSM-001**: The system shall encode incoming text using X-Encoder to produce query embeddings.
- **FR-SSM-002**: The system shall maintain a sliding window buffer of the last N turn embeddings (configurable, default 5).
- **FR-SSM-003**: The system shall calculate semantic variance as `Var(||e_i - centroid||)` for embeddings in the window.
- **FR-SSM-004**: The system shall apply EMA smoothing to variance values with configurable smoothing factor (default 0.3).
- **FR-SSM-005**: The system shall compare smoothed variance against threshold θ to detect boundary crossings.
- **FR-SSM-006**: The system shall emit one of four action states based on variance level: CONTINUE, PREPARE, RETRIEVE, RETRIEVE_CONSOLIDATE.
- **FR-SSM-007**: The system shall expose current semantic state for observability purposes.

### User Stories

- **US-SSM-001**: As the PRIME orchestrator, I want to know when a semantic boundary is crossed so that I can trigger retrieval selectively.
- **US-SSM-002**: As a developer, I want to configure variance thresholds so that I can tune sensitivity for different use cases.
- **US-SSM-003**: As an operator, I want to observe variance values over time so that I can debug retrieval behavior.
- **US-SSM-004**: As the caching system, I want advance notice (PREPARE state) so that I can pre-warm context caches.

### Business Rules and Constraints

- **BR-SSM-001**: Variance calculation MUST use Ward distance (L2 norm from centroid).
- **BR-SSM-002**: Window buffer MUST be FIFO (oldest embedding removed when full).
- **BR-SSM-003**: First `window_size` turns MUST return CONTINUE (insufficient data for variance).
- **BR-SSM-004**: EMA smoothing MUST use formula: `smoothed = α * current + (1-α) * previous`.

---

## 3. Non-Functional Requirements

### Performance Targets

| Metric | Target | Constraint |
|--------|--------|------------|
| Variance Calculation Latency | <5ms p50 | <10ms p95 |
| Memory Footprint | <50MB | For 1000-turn buffer |
| Throughput | >1000 updates/sec | Single instance |

### Security Requirements

- **SEC-SSM-001**: Embeddings MUST NOT be logged in production (contain semantic information).
- **SEC-SSM-002**: Configuration MUST be read-only during operation (prevent threshold manipulation).

### Scalability Considerations

- Stateful component (maintains embedding buffer per conversation)
- Scale horizontally by conversation ID partitioning
- No shared state between instances

---

## 4. Features & Flows

### Feature Breakdown

| Feature | Priority | Description |
|---------|----------|-------------|
| Variance Calculation | P0 | Core boundary detection algorithm |
| EMA Smoothing | P0 | Noise reduction for stable triggering |
| Action State Machine | P0 | CONTINUE/PREPARE/RETRIEVE/RETRIEVE_CONSOLIDATE |
| Window Buffer | P0 | Sliding window embedding storage |
| Configuration | P1 | Runtime-configurable thresholds |
| State Export | P2 | Observability/debugging interface |

### Key User Flows

**Flow 1: Update Semantic State**

```
Input: User message text
  │
  ▼
┌─────────────────┐
│ X-Encoder       │ ──▶ Query Embedding (1024-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Buffer Update   │ ──▶ Add to sliding window
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Centroid Calc   │ ──▶ Mean of window embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Variance Calc   │ ──▶ Var(||e_i - centroid||)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EMA Smoothing   │ ──▶ Smoothed variance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Threshold Check │ ──▶ Action State
└─────────────────┘

Output: SemanticStateUpdate {
  variance: float,
  smoothed_variance: float,
  action: ActionState,
  boundary_crossed: bool,
  embedding: ndarray
}
```

### Input/Output Specifications

**Input: Text string**
- Type: `str`
- Constraints: Non-empty, max 8192 characters

**Output: SemanticStateUpdate**
```python
class SemanticStateUpdate(BaseModel):
    """Result of semantic state update."""

    variance: float = Field(ge=0.0, description="Raw variance value")
    smoothed_variance: float = Field(ge=0.0, description="EMA-smoothed variance")
    action: ActionState = Field(description="Recommended action")
    boundary_crossed: bool = Field(description="True if variance > threshold")
    embedding: list[float] = Field(description="Query embedding for downstream use")
    window_size: int = Field(ge=1, description="Current window size")
    turn_number: int = Field(ge=0, description="Conversation turn number")
```

**ActionState Enum:**
```python
class ActionState(str, Enum):
    CONTINUE = "continue"           # variance < 0.5θ
    PREPARE = "prepare"             # 0.5θ ≤ variance < θ
    RETRIEVE = "retrieve"           # θ ≤ variance < 2θ
    RETRIEVE_CONSOLIDATE = "retrieve_consolidate"  # variance ≥ 2θ
```

---

## 5. Code Pattern Requirements

### Naming Conventions

- **Classes**: PascalCase (`SemanticStateMonitor`, `SemanticStateUpdate`)
- **Functions**: snake_case (`update`, `calculate_variance`, `get_action_state`)
- **Variables**: snake_case (`smoothed_variance`, `window_buffer`)
- **Constants**: SCREAMING_SNAKE_CASE (`DEFAULT_WINDOW_SIZE`, `DEFAULT_THRESHOLD`)

### Type Safety Requirements

- **Type hint coverage**: 100% for public API
- **Union syntax**: Use `|` operator (`float | None`)
- **Generics**: Use builtin generics (`list[float]`, `dict[str, Any]`)
- **Null handling**: Explicit (`T | None`)
- **Required import**: `from __future__ import annotations`

### Testing Approach

- **Framework**: pytest + pytest-asyncio
- **Coverage requirement**: ≥90%
- **Test patterns**: Arrange-Act-Assert (AAA)
- **Fixtures**: `@pytest.fixture` for SSM instances with default config

**Required Test Cases:**
- `test_boundary_detection_on_topic_change`
- `test_no_boundary_on_same_topic`
- `test_ema_smoothing_reduces_noise`
- `test_action_state_transitions`
- `test_window_buffer_fifo`
- `test_insufficient_data_returns_continue`

### Error Handling

- **Strategy**: Explicit raises (no silent failures)
- **Custom exceptions**: `SSMError`, `EncodingError`
- **Validation**: Input validation at public method boundaries

### Architecture Patterns

- **Module system**: Single module `src/prime/core/ssm.py`
- **Export style**: Named exports
- **Protocol**: Implement `Encoder` protocol for X-Encoder dependency injection

---

## 6. Acceptance Criteria

### Definition of Done

- [ ] All functional requirements implemented
- [ ] All non-functional requirements met
- [ ] Unit tests passing with ≥90% coverage
- [ ] Integration test with mock X-Encoder passing
- [ ] Type checking passes (`mypy --strict`)
- [ ] Linting passes (`ruff check`)
- [ ] Documentation complete (docstrings, API docs)
- [ ] Performance benchmarks meet targets

### Validation Approach

1. **Unit Testing**: pytest with parametrized test cases
2. **Integration Testing**: SSM + real X-Encoder on conversation samples
3. **Benchmark Testing**: Latency profiling with `timeit`
4. **Manual Evaluation**: Boundary detection accuracy on labeled dataset

---

## 7. Dependencies

### Technical Assumptions

- X-Encoder produces 1024-dimensional embeddings
- NumPy available for vector operations
- Pydantic v2 for data validation

### External Integrations

| Integration | Type | Purpose |
|-------------|------|---------|
| X-Encoder | Required | Query embedding generation |
| Sentence-Transformers | Required | Pre-trained encoder models |
| NumPy | Required | Vector math operations |

### Related Components

| Component | Relationship |
|-----------|--------------|
| MCS | Downstream consumer (receives RETRIEVE signal) |
| Predictor | Downstream consumer (receives boundary trigger) |
| PRIME | Parent orchestrator |

---

## 8. Configuration Schema

```python
class SSMConfig(BaseModel):
    """Configuration for Semantic State Monitor."""

    window_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of turns in sliding window"
    )
    variance_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Threshold θ for boundary detection"
    )
    smoothing_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="EMA smoothing coefficient α"
    )
    prepare_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of θ for PREPARE state trigger"
    )
    x_encoder_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for X-Encoder"
    )

    model_config = {"frozen": True}
```

---

## Appendix: Source Traceability

| Requirement | Source Document | Section |
|-------------|-----------------|---------|
| FR-SSM-001 | PRIME-Project-Overview.md | 4.1.2 Architecture |
| FR-SSM-003 | PRIME-Project-Overview.md | 4.1.4 Variance Calculation |
| FR-SSM-006 | PRIME-Project-Overview.md | 4.1.5 Action States |
| Performance | enhancement.md | Performance Optimization |
| Retrieval Reduction | strategic-intel.md | Technical Best Practices |
