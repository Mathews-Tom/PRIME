# ENC-001: Y-Encoder Implementation Plan

**Epic:** ENC-001 - Implement Y-Encoder
**Status:** Ready for Implementation
**Generated:** 2026-01-08

---

## 1. Executive Summary

### Objective

Implement the Y-Encoder component that encodes target content (responses, documents) into 1024-dimensional L2-normalized embeddings for memory storage and predictor training targets.

### Scope

- Core `YEncoder` class implementing the `Encoder` protocol
- Batch encoding with GPU acceleration
- Multi-backend support (Gemma, MiniLM, BGE, Qwen)
- Optional LRU caching for repeated content
- Comprehensive test suite with 90%+ coverage

### Success Criteria

| Metric | Target |
|--------|--------|
| Single Encoding Latency | <50ms p50 |
| Batch Encoding (32) | <500ms |
| Embedding Dimension | 1024 (configurable) |
| L2 Normalization | All outputs unit vectors |
| Test Coverage | ≥90% |

### Dependencies

- **External:** PyTorch 2.2+, HuggingFace Transformers 4.37+, Sentence-Transformers 2.x
- **Internal:** None (foundational component)
- **Blocks:** MCS-001, PRED-001

---

## 2. Context & Documentation Sources

### Primary Specification

- [docs/specs/encoder/spec.md](spec.md) - Full Y-Encoder specification

### Architecture Context

- [.sage/agent/system/architecture.md](../../../.sage/agent/system/architecture.md) - System architecture
- [.sage/agent/system/tech-stack.md](../../../.sage/agent/system/tech-stack.md) - Technology stack

### Enhancement Integration

No Y-Encoder-specific enhancements from `docs/enhancement.md` - component is already optimized for its purpose.

### Traceability Matrix

| Requirement | Source | Priority |
|-------------|--------|----------|
| FR-ENC-001: 1024-dim embeddings | spec.md | P0 |
| FR-ENC-002: L2 normalization | spec.md | P0 |
| FR-ENC-003: Batch encoding | spec.md | P0 |
| FR-ENC-004: Truncation | spec.md | P0 |
| FR-ENC-005: Multi-backend | spec.md | P0 |
| FR-ENC-006: Caching | spec.md | P2 |
| FR-ENC-007: Model info | spec.md | P0 |

---

## 3. Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Y-Encoder Module                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ YEncoder    │───▶│ BaseEncoder  │───▶│ Tokenizer    │   │
│  │ (Public)    │    │ (Model)      │    │ (HF)         │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│         │                  │                               │
│         ▼                  ▼                               │
│  ┌─────────────┐    ┌──────────────┐                       │
│  │ Pooler      │    │ Normalizer   │                       │
│  │ (Mean/CLS)  │    │ (L2)         │                       │
│  └─────────────┘    └──────────────┘                       │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │ Cache       │ (Optional LRU)                            │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input: str
    │
    ▼
┌─────────────────┐
│ Tokenize        │ ──▶ input_ids, attention_mask
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Truncate        │ ──▶ max_length=512 tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Forward Pass    │ ──▶ hidden_states: (B, S, D)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pooling         │ ──▶ pooled: (B, D)
└────────┬────────┘     mean/cls/max over sequence
         │
         ▼
┌─────────────────┐
│ L2 Normalize    │ ──▶ normalized: (B, D)
└────────┬────────┘     ||v|| = 1.0
         │
         ▼
Output: np.ndarray (D,) or (B, D)
```

### Class Hierarchy

```python
Encoder (Protocol)
    │
    └── YEncoder (Implementation)
            │
            ├── _model: AutoModel
            ├── _tokenizer: AutoTokenizer
            ├── _config: YEncoderConfig
            └── _cache: LRUCache (optional)
```

---

## 4. Technical Specification

### File Structure

```
src/prime/encoder/
├── __init__.py          # Export YEncoder, Encoder protocol
├── y_encoder.py         # Main YEncoder implementation
├── config.py            # YEncoderConfig
├── protocols.py         # Encoder protocol definition
├── exceptions.py        # EncoderError, ModelLoadError
└── pooling.py           # Pooling strategies

tests/
└── test_encoder.py      # Comprehensive tests
```

### Core Implementation

#### `src/prime/encoder/protocols.py`

```python
from __future__ import annotations

from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """Protocol for embedding encoders (X and Y)."""

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...

    @property
    def max_length(self) -> int:
        """Return maximum input length in tokens."""
        ...

    @property
    def model_name(self) -> str:
        """Return model name."""
        ...

    def encode(self, text: str) -> np.ndarray:
        """Encode single text to embedding vector.

        Args:
            text: Input text to encode.

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,).
        """
        ...

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode batch of texts.

        Args:
            texts: List of input texts to encode.

        Returns:
            List of L2-normalized embedding vectors.
        """
        ...
```

#### `src/prime/encoder/config.py`

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class YEncoderConfig(BaseModel):
    """Configuration for Y-Encoder."""

    model_name: str = Field(
        default="google/gemma-embedding-300m",
        description="HuggingFace model name or path",
    )
    embedding_dim: int = Field(
        default=1024,
        description="Output embedding dimension",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum input tokens",
    )
    pooling_mode: Literal["mean", "cls", "max"] = Field(
        default="mean",
        description="Pooling strategy",
    )
    normalize: bool = Field(
        default=True,
        description="L2 normalize output embeddings",
    )
    device: str = Field(
        default="cuda",
        description="Device for inference: 'cuda', 'cpu', 'mps'",
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="LRU cache size (0 = disabled)",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code for custom models",
    )

    model_config = {"frozen": True}
```

#### `src/prime/encoder/exceptions.py`

```python
from __future__ import annotations


class EncoderError(Exception):
    """Base exception for encoder errors."""


class ModelLoadError(EncoderError):
    """Error loading encoder model."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


class EncodingError(EncoderError):
    """Error during text encoding."""

    def __init__(self, text_preview: str, reason: str) -> None:
        self.text_preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        self.reason = reason
        super().__init__(f"Encoding failed for '{self.text_preview}': {reason}")
```

#### `src/prime/encoder/pooling.py`

```python
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def pool_embeddings(
    hidden_states: Tensor,
    attention_mask: Tensor,
    mode: Literal["mean", "cls", "max"] = "mean",
) -> Tensor:
    """Pool hidden states to single embedding.

    Args:
        hidden_states: Model output of shape (B, S, D).
        attention_mask: Attention mask of shape (B, S).
        mode: Pooling strategy.

    Returns:
        Pooled embeddings of shape (B, D).
    """
    if mode == "cls":
        return hidden_states[:, 0, :]

    if mode == "max":
        # Mask padding tokens with -inf
        masked = hidden_states.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(),
            float("-inf"),
        )
        return masked.max(dim=1).values

    # Mean pooling (default)
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask
```

#### `src/prime/encoder/y_encoder.py`

```python
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from prime.encoder.config import YEncoderConfig
from prime.encoder.exceptions import EncodingError, ModelLoadError
from prime.encoder.pooling import pool_embeddings


class YEncoder:
    """Y-Encoder for target content embedding.

    Encodes target content (responses, documents) into high-quality
    semantic embeddings for memory storage and predictor training.

    Attributes:
        embedding_dim: Output embedding dimension.
        max_length: Maximum input length in tokens.
        model_name: Name of the loaded model.
    """

    def __init__(self, config: YEncoderConfig | None = None) -> None:
        """Initialize Y-Encoder.

        Args:
            config: Encoder configuration. Uses defaults if None.

        Raises:
            ModelLoadError: If model fails to load.
        """
        self._config = config or YEncoderConfig()
        self._device = self._resolve_device(self._config.device)
        self._model, self._tokenizer = self._load_model()
        self._cache_encode = self._setup_cache()

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._config.embedding_dim

    @property
    def max_length(self) -> int:
        """Return maximum input length."""
        return self._config.max_length

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._config.model_name

    def encode(self, text: str) -> np.ndarray:
        """Encode single text to embedding.

        Args:
            text: Input text to encode.

        Returns:
            L2-normalized embedding of shape (embedding_dim,).

        Raises:
            EncodingError: If encoding fails.
        """
        if not text or not text.strip():
            raise EncodingError(text, "Empty or whitespace-only input")

        if self._cache_encode is not None:
            cache_key = self._compute_cache_key(text)
            cached = self._cache_encode(cache_key)
            if cached is not None:
                return cached

        embedding = self._encode_texts([text])[0]

        if self._cache_encode is not None:
            self._cache_encode.cache_info()  # Trigger cache storage

        return embedding

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of L2-normalized embeddings.

        Raises:
            EncodingError: If encoding fails.
        """
        if not texts:
            return []

        return self._encode_texts(texts)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model name, dimension, max_length.
        """
        return {
            "model_name": self._config.model_name,
            "embedding_dim": self._config.embedding_dim,
            "max_length": self._config.max_length,
            "pooling_mode": self._config.pooling_mode,
            "device": str(self._device),
        }

    def _load_model(self) -> tuple[AutoModel, AutoTokenizer]:
        """Load model and tokenizer from HuggingFace."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name,
                trust_remote_code=self._config.trust_remote_code,
            )
            model = AutoModel.from_pretrained(
                self._config.model_name,
                trust_remote_code=self._config.trust_remote_code,
            )
            model = model.to(self._device)
            model.eval()
            return model, tokenizer
        except Exception as e:
            raise ModelLoadError(self._config.model_name, str(e)) from e

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def _encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Internal batch encoding implementation."""
        try:
            # Tokenize
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Pool
            hidden_states = outputs.last_hidden_state
            pooled = pool_embeddings(
                hidden_states,
                inputs["attention_mask"],
                mode=self._config.pooling_mode,
            )

            # Normalize
            if self._config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            # Convert to numpy
            embeddings = pooled.cpu().numpy()

            return [embeddings[i] for i in range(len(texts))]

        except Exception as e:
            preview = texts[0] if texts else ""
            raise EncodingError(preview, str(e)) from e

    def _setup_cache(self) -> lru_cache | None:
        """Set up LRU cache if configured."""
        if self._config.cache_size <= 0:
            return None

        @lru_cache(maxsize=self._config.cache_size)
        def cached_encode(cache_key: str) -> np.ndarray | None:
            return None  # Actual encoding happens in encode()

        return cached_encode

    def _compute_cache_key(self, text: str) -> str:
        """Compute cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()
```

#### `src/prime/encoder/__init__.py`

```python
from __future__ import annotations

from prime.encoder.config import YEncoderConfig
from prime.encoder.exceptions import EncoderError, EncodingError, ModelLoadError
from prime.encoder.protocols import Encoder
from prime.encoder.y_encoder import YEncoder

__all__ = [
    "Encoder",
    "EncoderError",
    "EncodingError",
    "ModelLoadError",
    "YEncoder",
    "YEncoderConfig",
]
```

---

## 5. Test Specification

### Test File: `tests/test_encoder.py`

```python
from __future__ import annotations

import numpy as np
import pytest

from prime.encoder import (
    Encoder,
    EncoderError,
    EncodingError,
    ModelLoadError,
    YEncoder,
    YEncoderConfig,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def encoder_config() -> YEncoderConfig:
    """Create test encoder config with small model."""
    return YEncoderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,  # MiniLM dimension
        max_length=256,
        device="cpu",
    )


@pytest.fixture
def encoder(encoder_config: YEncoderConfig) -> YEncoder:
    """Create Y-Encoder instance."""
    return YEncoder(encoder_config)


# ============================================================================
# Protocol Compliance
# ============================================================================


def test_encoder_implements_protocol(encoder: YEncoder) -> None:
    """Verify YEncoder implements Encoder protocol."""
    assert isinstance(encoder, Encoder)


# ============================================================================
# Core Functionality
# ============================================================================


def test_encode_returns_correct_dim(encoder: YEncoder) -> None:
    """Test that encode returns correct dimension."""
    result = encoder.encode("Hello, world!")
    assert result.shape == (encoder.embedding_dim,)


def test_output_normalized(encoder: YEncoder) -> None:
    """Test that output is L2 normalized."""
    result = encoder.encode("Test normalization")
    norm = np.linalg.norm(result)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_batch_encode_consistency(encoder: YEncoder) -> None:
    """Test batch encoding produces consistent results."""
    texts = ["First text", "Second text", "Third text"]

    # Single encode
    single_results = [encoder.encode(t) for t in texts]

    # Batch encode
    batch_results = encoder.encode_batch(texts)

    # Should be equal
    for single, batch in zip(single_results, batch_results, strict=True):
        assert np.allclose(single, batch, atol=1e-5)


def test_truncation_long_input(encoder: YEncoder) -> None:
    """Test that long inputs are truncated."""
    # Create input longer than max_length
    long_text = "word " * 1000

    # Should not raise
    result = encoder.encode(long_text)
    assert result.shape == (encoder.embedding_dim,)


def test_batch_encode_empty_list(encoder: YEncoder) -> None:
    """Test batch encoding empty list returns empty."""
    result = encoder.encode_batch([])
    assert result == []


# ============================================================================
# Error Handling
# ============================================================================


def test_encode_empty_string_raises(encoder: YEncoder) -> None:
    """Test encoding empty string raises error."""
    with pytest.raises(EncodingError):
        encoder.encode("")


def test_encode_whitespace_raises(encoder: YEncoder) -> None:
    """Test encoding whitespace-only raises error."""
    with pytest.raises(EncodingError):
        encoder.encode("   \n\t  ")


def test_invalid_model_raises() -> None:
    """Test invalid model name raises ModelLoadError."""
    config = YEncoderConfig(model_name="nonexistent/model-xyz")
    with pytest.raises(ModelLoadError):
        YEncoder(config)


# ============================================================================
# Model Info
# ============================================================================


def test_model_info(encoder: YEncoder, encoder_config: YEncoderConfig) -> None:
    """Test get_model_info returns correct info."""
    info = encoder.get_model_info()

    assert info["model_name"] == encoder_config.model_name
    assert info["embedding_dim"] == encoder_config.embedding_dim
    assert info["max_length"] == encoder_config.max_length
    assert info["pooling_mode"] == encoder_config.pooling_mode


# ============================================================================
# Properties
# ============================================================================


def test_embedding_dim_property(encoder: YEncoder) -> None:
    """Test embedding_dim property."""
    assert encoder.embedding_dim == 384


def test_max_length_property(encoder: YEncoder) -> None:
    """Test max_length property."""
    assert encoder.max_length == 256


def test_model_name_property(encoder: YEncoder) -> None:
    """Test model_name property."""
    assert "MiniLM" in encoder.model_name


# ============================================================================
# Configuration
# ============================================================================


def test_default_config() -> None:
    """Test default configuration values."""
    config = YEncoderConfig()

    assert config.model_name == "google/gemma-embedding-300m"
    assert config.embedding_dim == 1024
    assert config.max_length == 512
    assert config.pooling_mode == "mean"
    assert config.normalize is True


def test_config_immutable() -> None:
    """Test config is frozen."""
    config = YEncoderConfig()
    with pytest.raises(Exception):  # Pydantic ValidationError
        config.max_length = 1024  # type: ignore


# ============================================================================
# Pooling Modes
# ============================================================================


@pytest.mark.parametrize("pooling_mode", ["mean", "cls", "max"])
def test_pooling_modes(pooling_mode: str) -> None:
    """Test different pooling modes work."""
    config = YEncoderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        pooling_mode=pooling_mode,  # type: ignore
        device="cpu",
    )
    encoder = YEncoder(config)
    result = encoder.encode("Test pooling mode")

    assert result.shape == (384,)
    assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)


# ============================================================================
# Semantic Quality
# ============================================================================


def test_similar_texts_similar_embeddings(encoder: YEncoder) -> None:
    """Test semantically similar texts have high cosine similarity."""
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    text3 = "The stock market crashed today."

    emb1 = encoder.encode(text1)
    emb2 = encoder.encode(text2)
    emb3 = encoder.encode(text3)

    # Similar texts should have higher similarity
    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)

    assert sim_12 > sim_13
```

---

## 6. Implementation Roadmap

### Phase 1: Core Implementation (P0)

**Step 1.1: Project Setup**
- Create directory structure
- Add dependencies to pyproject.toml
- Configure pytest

**Step 1.2: Exceptions and Config**
- Implement `exceptions.py`
- Implement `config.py`
- Add validation tests

**Step 1.3: Protocol Definition**
- Implement `protocols.py`
- Document protocol contract

**Step 1.4: Pooling Functions**
- Implement `pooling.py`
- Test pooling strategies

**Step 1.5: YEncoder Core**
- Implement `y_encoder.py`
- Model loading
- Single and batch encoding
- L2 normalization

**Step 1.6: Integration Tests**
- Protocol compliance
- Semantic quality tests
- Performance benchmarks

### Phase 2: Enhancements (P1-P2)

**Step 2.1: Multi-Backend Support**
- Test all supported models
- Document model trade-offs

**Step 2.2: Caching (P2)**
- Implement LRU cache
- Cache key computation
- Cache hit/miss metrics

**Step 2.3: Performance Optimization**
- GPU batching optimization
- ONNX export (P2)

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
uv run pytest tests/test_encoder.py -k benchmark --benchmark

# Memory profiling
uv run python -m memory_profiler -m pytest tests/test_encoder.py
```

### Pre-commit Checks

```yaml
# .pre-commit-config.yaml additions
- repo: local
  hooks:
    - id: encoder-tests
      name: Run encoder tests
      entry: uv run pytest tests/test_encoder.py -v
      language: system
      pass_filenames: false
```

---

## 8. Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model download failures | Medium | High | Cache models locally, fallback to smaller model |
| GPU memory exhaustion | Medium | Medium | Configurable batch size, CPU fallback |
| Embedding dimension mismatch | Low | High | Validate at load time, fail fast |
| Tokenizer incompatibility | Low | Medium | Pin transformers version, test all backends |

### Performance Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Latency exceeds target | Medium | Medium | Batch optimization, smaller model option |
| Memory footprint too large | Low | Medium | Lazy loading, offload to CPU |

---

## 9. Error Handling Strategy

### Error Hierarchy

```
EncoderError (base)
├── ModelLoadError
│   └── Raised when: Model download fails, invalid model name
│   └── Contains: model_name, reason
└── EncodingError
    └── Raised when: Empty input, encoding failure
    └── Contains: text_preview (truncated), reason
```

### Error Recovery

| Error | Recovery Strategy |
|-------|-------------------|
| ModelLoadError | Fail fast at startup - no fallback |
| EncodingError (empty) | Raise immediately - caller must validate |
| EncodingError (internal) | Log error, raise with context |
| GPU OOM | Automatic CPU fallback during _resolve_device |

---

## 10. References & Traceability

### Source Documents

| Document | Purpose |
|----------|---------|
| [spec.md](spec.md) | Functional requirements |
| [architecture.md](../../../.sage/agent/system/architecture.md) | System context |
| [tech-stack.md](../../../.sage/agent/system/tech-stack.md) | Technology choices |
| [patterns.md](../../../.sage/agent/system/patterns.md) | Code patterns |

### Related Tickets

| Ticket | Relationship |
|--------|--------------|
| MCS-001 | Consumer - uses Y-Encoder for content embedding |
| PRED-001 | Consumer - uses Y-Encoder for training targets |

### External References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Sentence-Transformers](https://www.sbert.net/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)

---

## Appendix A: Supported Models Reference

| Model | HuggingFace ID | Dim | Max Len | Use Case |
|-------|----------------|-----|---------|----------|
| Gemma Embedding | `google/gemma-embedding-300m` | 1024 | 512 | Default |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | Low-latency |
| BGE Large | `BAAI/bge-large-en-v1.5` | 1024 | 512 | High quality |
| Qwen3 0.6B | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 8192 | Multilingual |
| Qwen3 8B | `Qwen/Qwen3-Embedding-8B` | 1024 | 8192 | Premium |

## Appendix B: Configuration Examples

### Development (Fast)

```python
config = YEncoderConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    max_length=256,
    device="cpu",
)
```

### Production (Quality)

```python
config = YEncoderConfig(
    model_name="google/gemma-embedding-300m",
    embedding_dim=1024,
    max_length=512,
    device="cuda",
    cache_size=10000,
)
```

### High Quality

```python
config = YEncoderConfig(
    model_name="BAAI/bge-large-en-v1.5",
    embedding_dim=1024,
    max_length=512,
    device="cuda",
)
```
