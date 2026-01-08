# SSM-001: Semantic State Monitor Implementation Plan

**Epic:** SSM-001 - Implement Semantic State Monitor
**Status:** Ready for Implementation
**Generated:** 2026-01-08

---

## 1. Executive Summary

### Objective

Implement the Semantic State Monitor (SSM) - PRIME's intelligent retrieval trigger that monitors semantic trajectories and triggers retrieval only when significant semantic boundaries are crossed, reducing unnecessary retrievals by 60-70%.

### Scope

- Core `SemanticStateMonitor` class with sliding window buffer
- Variance-based boundary detection using Ward distance
- EMA smoothing for stable triggering
- Four-state action machine (CONTINUE, PREPARE, RETRIEVE, RETRIEVE_CONSOLIDATE)
- X-Encoder integration for query embedding
- Comprehensive test suite with 90%+ coverage

### Success Criteria

| Metric | Target |
|--------|--------|
| Retrieval Reduction | 60-70% vs baseline |
| Boundary Detection Precision | >85% |
| Boundary Detection Recall | >90% |
| Variance Calculation Latency | <5ms p50 |
| Test Coverage | ≥90% |

### Dependencies

- **External:** NumPy, Sentence-Transformers, Pydantic v2
- **Internal:** X-Encoder (uses Encoder protocol from ENC-001)
- **Blocks:** API-001

---

## 2. Context & Documentation Sources

### Primary Specification

- [docs/specs/ssm/spec.md](spec.md) - Full SSM specification

### Architecture Context

- [.sage/agent/system/architecture.md](../../../.sage/agent/system/architecture.md) - System architecture
- [.sage/agent/system/tech-stack.md](../../../.sage/agent/system/tech-stack.md) - Technology stack

### Enhancement Integration

No SSM-specific enhancements - the variance-based triggering is the core innovation.

### Traceability Matrix

| Requirement | Source | Priority |
|-------------|--------|----------|
| FR-SSM-001: X-Encoder embedding | spec.md | P0 |
| FR-SSM-002: Sliding window buffer | spec.md | P0 |
| FR-SSM-003: Ward variance calculation | spec.md | P0 |
| FR-SSM-004: EMA smoothing | spec.md | P0 |
| FR-SSM-005: Threshold comparison | spec.md | P0 |
| FR-SSM-006: Action states | spec.md | P0 |
| FR-SSM-007: State observability | spec.md | P2 |

---

## 3. Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 Semantic State Monitor                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ SSM         │───▶│ X-Encoder    │───▶│ Embedding    │   │
│  │ (Public)    │    │ (Protocol)   │    │ Buffer       │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│         │                                     │             │
│         ▼                                     ▼             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Variance    │◀───│ Centroid     │◀───│ Window       │   │
│  │ Calculator  │    │ Calculator   │    │ Manager      │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐    ┌──────────────┐                       │
│  │ EMA         │───▶│ Action State │                       │
│  │ Smoother    │    │ Machine      │                       │
│  └─────────────┘    └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### State Machine

```
                    variance < 0.5θ
    ┌────────────────────────────────────────┐
    │                                        │
    ▼                                        │
┌─────────┐     0.5θ ≤ v < θ     ┌─────────┐
│CONTINUE │─────────────────────▶│ PREPARE │
└─────────┘                      └─────────┘
    │                                │
    │ θ ≤ v < 2θ                     │ θ ≤ v < 2θ
    │                                │
    ▼                                ▼
┌─────────┐     variance ≥ 2θ    ┌──────────────────┐
│RETRIEVE │─────────────────────▶│RETRIEVE_CONSOLIDATE│
└─────────┘                      └──────────────────┘
```

### Data Flow

```
Input: User message text
    │
    ▼
┌─────────────────┐
│ X-Encoder       │ ──▶ Query Embedding S_X (1024-dim)
│ (Injected)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Window Buffer   │ ──▶ FIFO append (oldest removed if full)
│ [e₁, e₂, ..eₙ]  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Centroid        │ ──▶ μ = mean(window)
│ Calculation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Variance Calc   │ ──▶ Var(||eᵢ - μ||) Ward distance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EMA Smoothing   │ ──▶ s = α·v + (1-α)·s_prev
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Action State    │ ──▶ Compare smoothed vs thresholds
│ Determination   │
└────────┬────────┘
         │
         ▼
Output: SemanticStateUpdate
```

---

## 4. Technical Specification

### File Structure

```
src/prime/core/
├── __init__.py          # Export SSM, ActionState
├── ssm.py               # SemanticStateMonitor implementation
├── ssm_config.py        # SSMConfig
└── ssm_types.py         # SemanticStateUpdate, ActionState

tests/
└── test_ssm.py          # Comprehensive tests
```

### Core Implementation

#### `src/prime/core/ssm_types.py`

```python
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActionState(str, Enum):
    """Action states emitted by SSM based on variance level."""

    CONTINUE = "continue"  # variance < 0.5θ - No retrieval needed
    PREPARE = "prepare"  # 0.5θ ≤ variance < θ - Pre-warm caches
    RETRIEVE = "retrieve"  # θ ≤ variance < 2θ - Trigger retrieval
    RETRIEVE_CONSOLIDATE = "retrieve_consolidate"  # variance ≥ 2θ - Retrieve + consolidate


class SemanticStateUpdate(BaseModel):
    """Result of semantic state update."""

    variance: float = Field(ge=0.0, description="Raw variance value")
    smoothed_variance: float = Field(ge=0.0, description="EMA-smoothed variance")
    action: ActionState = Field(description="Recommended action")
    boundary_crossed: bool = Field(description="True if smoothed variance > threshold")
    embedding: list[float] = Field(description="Query embedding for downstream use")
    window_size: int = Field(ge=1, description="Current window size")
    turn_number: int = Field(ge=0, description="Conversation turn number")

    model_config = {"frozen": True}
```

#### `src/prime/core/ssm_config.py`

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class SSMConfig(BaseModel):
    """Configuration for Semantic State Monitor."""

    window_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of turns in sliding window",
    )
    variance_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Threshold θ for boundary detection",
    )
    smoothing_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="EMA smoothing coefficient α",
    )
    prepare_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio of θ for PREPARE state trigger",
    )
    consolidate_ratio: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Ratio of θ for RETRIEVE_CONSOLIDATE trigger",
    )
    embedding_dim: int = Field(
        default=1024,
        ge=1,
        description="Expected embedding dimension",
    )

    model_config = {"frozen": True}
```

#### `src/prime/core/ssm.py`

```python
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from prime.core.ssm_config import SSMConfig
from prime.core.ssm_types import ActionState, SemanticStateUpdate
from prime.encoder import Encoder


class SSMError(Exception):
    """Base exception for SSM errors."""


class InsufficientDataError(SSMError):
    """Raised when window has insufficient data for variance calculation."""


class SemanticStateMonitor:
    """Semantic State Monitor for intelligent retrieval triggering.

    Monitors the semantic trajectory of conversations and triggers
    retrieval only when significant semantic boundaries are crossed.

    Attributes:
        window_size: Number of turns in the sliding window.
        variance_threshold: Threshold θ for boundary detection.
        smoothing_factor: EMA smoothing coefficient α.
    """

    def __init__(
        self,
        encoder: Encoder,
        config: SSMConfig | None = None,
    ) -> None:
        """Initialize SSM.

        Args:
            encoder: X-Encoder for query embedding (implements Encoder protocol).
            config: SSM configuration. Uses defaults if None.
        """
        self._encoder = encoder
        self._config = config or SSMConfig()
        self._buffer: deque[np.ndarray] = deque(maxlen=self._config.window_size)
        self._smoothed_variance: float = 0.0
        self._turn_number: int = 0

    @property
    def window_size(self) -> int:
        """Return window size."""
        return self._config.window_size

    @property
    def variance_threshold(self) -> float:
        """Return variance threshold θ."""
        return self._config.variance_threshold

    @property
    def smoothing_factor(self) -> float:
        """Return EMA smoothing factor α."""
        return self._config.smoothing_factor

    def update(self, text: str) -> SemanticStateUpdate:
        """Update semantic state with new text.

        Args:
            text: User message text to process.

        Returns:
            SemanticStateUpdate with variance, action, and embedding.

        Raises:
            SSMError: If encoding fails.
        """
        if not text or not text.strip():
            raise SSMError("Empty or whitespace-only input")

        # Encode text
        embedding = self._encoder.encode(text)

        # Validate embedding dimension
        if embedding.shape[0] != self._config.embedding_dim:
            raise SSMError(
                f"Embedding dimension mismatch: expected {self._config.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )

        # Update buffer (FIFO)
        self._buffer.append(embedding)
        self._turn_number += 1

        # Calculate variance
        raw_variance = self._calculate_variance()

        # Apply EMA smoothing
        self._smoothed_variance = self._apply_ema(raw_variance)

        # Determine action state
        action = self._determine_action(self._smoothed_variance)

        # Check boundary crossing
        boundary_crossed = self._smoothed_variance >= self._config.variance_threshold

        return SemanticStateUpdate(
            variance=raw_variance,
            smoothed_variance=self._smoothed_variance,
            action=action,
            boundary_crossed=boundary_crossed,
            embedding=embedding.tolist(),
            window_size=len(self._buffer),
            turn_number=self._turn_number,
        )

    def reset(self) -> None:
        """Reset SSM state for new conversation."""
        self._buffer.clear()
        self._smoothed_variance = 0.0
        self._turn_number = 0

    def get_state(self) -> dict[str, Any]:
        """Get current SSM state for observability.

        Returns:
            Dictionary with current state information.
        """
        return {
            "turn_number": self._turn_number,
            "window_size": len(self._buffer),
            "window_capacity": self._config.window_size,
            "smoothed_variance": self._smoothed_variance,
            "variance_threshold": self._config.variance_threshold,
            "last_action": self._determine_action(self._smoothed_variance).value,
        }

    def _calculate_variance(self) -> float:
        """Calculate Ward variance of embeddings in window.

        Ward variance: Var(||e_i - centroid||)
        """
        if len(self._buffer) < 2:
            # Insufficient data - return 0
            return 0.0

        # Stack embeddings
        embeddings = np.stack(list(self._buffer))

        # Calculate centroid (mean)
        centroid = embeddings.mean(axis=0)

        # Calculate L2 distances from centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)

        # Return variance of distances
        return float(np.var(distances))

    def _apply_ema(self, current: float) -> float:
        """Apply exponential moving average smoothing.

        Formula: smoothed = α * current + (1-α) * previous
        """
        alpha = self._config.smoothing_factor

        if self._turn_number == 1:
            # First turn - no previous value
            return current

        return alpha * current + (1 - alpha) * self._smoothed_variance

    def _determine_action(self, smoothed_variance: float) -> ActionState:
        """Determine action state based on smoothed variance.

        States:
        - CONTINUE: variance < prepare_ratio * θ
        - PREPARE: prepare_ratio * θ ≤ variance < θ
        - RETRIEVE: θ ≤ variance < consolidate_ratio * θ
        - RETRIEVE_CONSOLIDATE: variance ≥ consolidate_ratio * θ
        """
        theta = self._config.variance_threshold
        prepare_threshold = self._config.prepare_ratio * theta
        consolidate_threshold = self._config.consolidate_ratio * theta

        if smoothed_variance < prepare_threshold:
            return ActionState.CONTINUE
        elif smoothed_variance < theta:
            return ActionState.PREPARE
        elif smoothed_variance < consolidate_threshold:
            return ActionState.RETRIEVE
        else:
            return ActionState.RETRIEVE_CONSOLIDATE
```

#### `src/prime/core/__init__.py`

```python
from __future__ import annotations

from prime.core.ssm import InsufficientDataError, SemanticStateMonitor, SSMError
from prime.core.ssm_config import SSMConfig
from prime.core.ssm_types import ActionState, SemanticStateUpdate

__all__ = [
    "ActionState",
    "InsufficientDataError",
    "SemanticStateMonitor",
    "SemanticStateUpdate",
    "SSMConfig",
    "SSMError",
]
```

---

## 5. Test Specification

### Test File: `tests/test_ssm.py`

```python
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from prime.core import (
    ActionState,
    SemanticStateMonitor,
    SemanticStateUpdate,
    SSMConfig,
    SSMError,
)
from prime.encoder import Encoder


# ============================================================================
# Mock Encoder
# ============================================================================


class MockEncoder:
    """Mock X-Encoder for testing."""

    def __init__(self, embedding_dim: int = 1024) -> None:
        self._embedding_dim = embedding_dim
        self._embeddings: list[np.ndarray] = []
        self._call_count = 0

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
        """Return predetermined or random embedding."""
        self._call_count += 1
        if self._embeddings:
            return self._embeddings.pop(0)
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(self._embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)  # Normalize

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]

    def set_embeddings(self, embeddings: list[np.ndarray]) -> None:
        """Set predetermined embeddings for testing."""
        self._embeddings = list(embeddings)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Create mock encoder."""
    return MockEncoder(embedding_dim=1024)


@pytest.fixture
def ssm_config() -> SSMConfig:
    """Create test SSM config."""
    return SSMConfig(
        window_size=5,
        variance_threshold=0.15,
        smoothing_factor=0.3,
        embedding_dim=1024,
    )


@pytest.fixture
def ssm(mock_encoder: MockEncoder, ssm_config: SSMConfig) -> SemanticStateMonitor:
    """Create SSM instance with mock encoder."""
    return SemanticStateMonitor(encoder=mock_encoder, config=ssm_config)


# ============================================================================
# Basic Functionality
# ============================================================================


def test_update_returns_semantic_state(ssm: SemanticStateMonitor) -> None:
    """Test that update returns SemanticStateUpdate."""
    result = ssm.update("Hello, how are you?")

    assert isinstance(result, SemanticStateUpdate)
    assert result.variance >= 0.0
    assert result.smoothed_variance >= 0.0
    assert isinstance(result.action, ActionState)
    assert isinstance(result.boundary_crossed, bool)
    assert len(result.embedding) == 1024
    assert result.turn_number == 1


def test_turn_number_increments(ssm: SemanticStateMonitor) -> None:
    """Test turn number increments with each update."""
    for i in range(5):
        result = ssm.update(f"Message {i}")
        assert result.turn_number == i + 1


def test_window_size_grows_then_caps(ssm: SemanticStateMonitor) -> None:
    """Test window size grows to max then stays constant."""
    for i in range(10):
        result = ssm.update(f"Message {i}")
        expected = min(i + 1, ssm.window_size)
        assert result.window_size == expected


# ============================================================================
# Boundary Detection
# ============================================================================


def test_boundary_detection_on_topic_change(mock_encoder: MockEncoder) -> None:
    """Test boundary is detected when topic changes."""
    # Create embeddings - first 4 similar, last one very different
    similar_emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    different_emb = np.array([0.0] * 512 + [1.0] + [0.0] * 511, dtype=np.float32)

    embeddings = [similar_emb.copy() for _ in range(4)]
    embeddings.append(different_emb)

    mock_encoder.set_embeddings(embeddings)

    ssm = SemanticStateMonitor(
        encoder=mock_encoder,
        config=SSMConfig(
            window_size=5,
            variance_threshold=0.01,  # Low threshold for test
            smoothing_factor=0.5,
            embedding_dim=1024,
        ),
    )

    # Process same-topic messages
    for i in range(4):
        result = ssm.update(f"Topic A message {i}")
        assert not result.boundary_crossed, f"Unexpected boundary at turn {i}"

    # Process different topic
    result = ssm.update("Topic B completely different")
    assert result.boundary_crossed


def test_no_boundary_on_same_topic(mock_encoder: MockEncoder) -> None:
    """Test no boundary detected for same topic."""
    # Create all similar embeddings
    base_emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    embeddings = [base_emb + np.random.randn(1024) * 0.01 for _ in range(10)]
    for emb in embeddings:
        emb /= np.linalg.norm(emb)

    mock_encoder.set_embeddings(embeddings)

    ssm = SemanticStateMonitor(
        encoder=mock_encoder,
        config=SSMConfig(
            window_size=5,
            variance_threshold=0.5,  # High threshold
            smoothing_factor=0.3,
            embedding_dim=1024,
        ),
    )

    # All messages should be CONTINUE (no boundary)
    for i in range(10):
        result = ssm.update(f"Same topic {i}")
        assert not result.boundary_crossed


# ============================================================================
# EMA Smoothing
# ============================================================================


def test_ema_smoothing_reduces_noise(mock_encoder: MockEncoder) -> None:
    """Test EMA smoothing reduces variance spikes."""
    # Create embeddings with one spike
    base_emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    spike_emb = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)

    embeddings = [base_emb.copy() for _ in range(3)]
    embeddings.append(spike_emb)  # Spike
    embeddings.append(base_emb.copy())  # Back to normal

    mock_encoder.set_embeddings(embeddings)

    ssm = SemanticStateMonitor(
        encoder=mock_encoder,
        config=SSMConfig(
            window_size=5,
            variance_threshold=0.5,
            smoothing_factor=0.3,  # Low alpha = more smoothing
            embedding_dim=1024,
        ),
    )

    variances = []
    smoothed_variances = []

    for i in range(5):
        result = ssm.update(f"Message {i}")
        variances.append(result.variance)
        smoothed_variances.append(result.smoothed_variance)

    # After spike, smoothed should be less than raw
    assert smoothed_variances[3] < variances[3] or variances[3] == 0


# ============================================================================
# Action States
# ============================================================================


def test_action_state_transitions(mock_encoder: MockEncoder) -> None:
    """Test action state transitions based on variance."""
    ssm = SemanticStateMonitor(
        encoder=mock_encoder,
        config=SSMConfig(
            window_size=5,
            variance_threshold=0.15,
            prepare_ratio=0.5,
            consolidate_ratio=2.0,
            embedding_dim=1024,
        ),
    )

    # Test initial state (insufficient data)
    result = ssm.update("First message")
    assert result.action == ActionState.CONTINUE  # No variance yet


def test_action_state_continue(ssm: SemanticStateMonitor) -> None:
    """Test CONTINUE action for low variance."""
    # Low variance should return CONTINUE
    result = ssm.update("Test")
    assert result.action in [ActionState.CONTINUE, ActionState.PREPARE]


# ============================================================================
# Window Buffer
# ============================================================================


def test_window_buffer_fifo(mock_encoder: MockEncoder) -> None:
    """Test window buffer is FIFO."""
    ssm = SemanticStateMonitor(
        encoder=mock_encoder,
        config=SSMConfig(window_size=3, embedding_dim=1024),
    )

    # Fill buffer
    for i in range(5):
        ssm.update(f"Message {i}")

    # Buffer should have last 3 embeddings
    assert len(ssm._buffer) == 3


def test_insufficient_data_returns_continue(ssm: SemanticStateMonitor) -> None:
    """Test insufficient data returns CONTINUE with zero variance."""
    result = ssm.update("Single message")

    assert result.action == ActionState.CONTINUE
    assert result.variance == 0.0  # Can't calculate variance with 1 point


# ============================================================================
# Reset
# ============================================================================


def test_reset_clears_state(ssm: SemanticStateMonitor) -> None:
    """Test reset clears all state."""
    # Populate state
    for i in range(10):
        ssm.update(f"Message {i}")

    assert ssm._turn_number == 10

    # Reset
    ssm.reset()

    assert ssm._turn_number == 0
    assert len(ssm._buffer) == 0
    assert ssm._smoothed_variance == 0.0


# ============================================================================
# Error Handling
# ============================================================================


def test_empty_input_raises(ssm: SemanticStateMonitor) -> None:
    """Test empty input raises SSMError."""
    with pytest.raises(SSMError):
        ssm.update("")


def test_whitespace_input_raises(ssm: SemanticStateMonitor) -> None:
    """Test whitespace-only input raises SSMError."""
    with pytest.raises(SSMError):
        ssm.update("   \n\t  ")


def test_embedding_dimension_mismatch_raises(mock_encoder: MockEncoder) -> None:
    """Test dimension mismatch raises SSMError."""
    # Create encoder with wrong dimension
    mock_encoder._embedding_dim = 512
    wrong_emb = np.random.randn(512).astype(np.float32)
    mock_encoder.set_embeddings([wrong_emb])

    ssm = SemanticStateMonitor(
        encoder=mock_encoder,
        config=SSMConfig(embedding_dim=1024),  # Expects 1024
    )

    with pytest.raises(SSMError, match="dimension mismatch"):
        ssm.update("Test")


# ============================================================================
# Observability
# ============================================================================


def test_get_state(ssm: SemanticStateMonitor) -> None:
    """Test get_state returns current state."""
    ssm.update("Test message")

    state = ssm.get_state()

    assert state["turn_number"] == 1
    assert state["window_size"] == 1
    assert state["window_capacity"] == 5
    assert "smoothed_variance" in state
    assert "variance_threshold" in state
    assert "last_action" in state


# ============================================================================
# Configuration
# ============================================================================


def test_default_config() -> None:
    """Test default configuration values."""
    config = SSMConfig()

    assert config.window_size == 5
    assert config.variance_threshold == 0.15
    assert config.smoothing_factor == 0.3


def test_config_immutable() -> None:
    """Test config is frozen."""
    config = SSMConfig()
    with pytest.raises(Exception):
        config.window_size = 10  # type: ignore


# ============================================================================
# Properties
# ============================================================================


def test_properties(ssm: SemanticStateMonitor, ssm_config: SSMConfig) -> None:
    """Test SSM properties match config."""
    assert ssm.window_size == ssm_config.window_size
    assert ssm.variance_threshold == ssm_config.variance_threshold
    assert ssm.smoothing_factor == ssm_config.smoothing_factor
```

---

## 6. Implementation Roadmap

### Phase 1: Core Implementation (P0)

**Step 1.1: Types and Config**
- Implement `ssm_types.py` (ActionState, SemanticStateUpdate)
- Implement `ssm_config.py` (SSMConfig)
- Add validation tests

**Step 1.2: Core SSM**
- Implement `ssm.py` core class
- Window buffer management (deque)
- Variance calculation (Ward distance)
- EMA smoothing

**Step 1.3: Action State Machine**
- Implement threshold-based state determination
- CONTINUE/PREPARE/RETRIEVE/RETRIEVE_CONSOLIDATE transitions

**Step 1.4: X-Encoder Integration**
- Accept Encoder protocol for dependency injection
- Validate embedding dimensions

**Step 1.5: Tests**
- Unit tests for all functionality
- Mock encoder for isolated testing
- Edge cases (empty input, dimension mismatch)

### Phase 2: Enhancements (P1-P2)

**Step 2.1: Observability (P2)**
- get_state() method for debugging
- Metrics export capability

**Step 2.2: Performance Optimization**
- NumPy vectorization
- Benchmark against latency targets

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
# Latency benchmark
uv run python -c "
import time
import numpy as np
from prime.core import SemanticStateMonitor, SSMConfig
from prime.encoder import YEncoder, YEncoderConfig

# Setup
encoder = YEncoder(YEncoderConfig(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    embedding_dim=384,
    device='cpu'
))
ssm = SemanticStateMonitor(encoder, SSMConfig(embedding_dim=384))

# Benchmark
times = []
for i in range(100):
    start = time.perf_counter()
    ssm.update(f'Test message {i}')
    times.append((time.perf_counter() - start) * 1000)

print(f'p50: {np.percentile(times, 50):.2f}ms')
print(f'p95: {np.percentile(times, 95):.2f}ms')
"
```

---

## 8. Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| X-Encoder latency dominating | High | Medium | SSM variance calc is O(n), encoder is the bottleneck |
| Variance threshold tuning | Medium | Medium | Configurable threshold, provide tuning guidance |
| Memory with large windows | Low | Low | Default window_size=5, max=20 |

### Algorithmic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| False positives (unnecessary retrieval) | Medium | Medium | EMA smoothing reduces spikes |
| False negatives (missed boundaries) | Medium | High | High recall target (90%), force_retrieval override |

---

## 9. Error Handling Strategy

### Error Hierarchy

```
SSMError (base)
├── InsufficientDataError (deprecated - return 0 variance instead)
└── Generic SSMError
    └── Raised when: Empty input, dimension mismatch
```

### Error Recovery

| Error | Recovery Strategy |
|-------|-------------------|
| SSMError (empty input) | Raise immediately - caller validates |
| Dimension mismatch | Raise immediately - configuration error |
| Encoder failure | Propagate EncoderError from X-Encoder |

---

## 10. References & Traceability

### Source Documents

| Document | Purpose |
|----------|---------|
| [spec.md](spec.md) | Functional requirements |
| [architecture.md](../../../.sage/agent/system/architecture.md) | System context |
| [tech-stack.md](../../../.sage/agent/system/tech-stack.md) | Technology choices |

### Related Tickets

| Ticket | Relationship |
|--------|--------------|
| ENC-001 | Provides Encoder protocol used for X-Encoder |
| MCS-001 | Receives RETRIEVE signal from SSM |
| PRED-001 | Receives boundary trigger from SSM |
| API-001 | Integrates SSM into PRIME orchestration |

---

## Appendix A: Variance Calculation Details

### Ward Distance Formula

```
Given window W = {e₁, e₂, ..., eₙ} of embeddings:

1. Centroid: μ = (1/n) * Σᵢ eᵢ

2. Distances: dᵢ = ||eᵢ - μ||₂

3. Variance: Var(W) = (1/n) * Σᵢ (dᵢ - d̄)²

Where d̄ = (1/n) * Σᵢ dᵢ
```

### EMA Smoothing Formula

```
Given:
- α = smoothing_factor (default 0.3)
- vₜ = current variance
- sₜ₋₁ = previous smoothed variance

Smoothed: sₜ = α * vₜ + (1-α) * sₜ₋₁
```

## Appendix B: Action State Thresholds

| State | Condition | Typical θ=0.15 |
|-------|-----------|----------------|
| CONTINUE | v < 0.5θ | v < 0.075 |
| PREPARE | 0.5θ ≤ v < θ | 0.075 ≤ v < 0.15 |
| RETRIEVE | θ ≤ v < 2θ | 0.15 ≤ v < 0.30 |
| RETRIEVE_CONSOLIDATE | v ≥ 2θ | v ≥ 0.30 |
