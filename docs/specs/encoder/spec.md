# Y-Encoder Specification

## 1. Overview

### Purpose and Business Value

The Y-Encoder encodes target content (responses, documents) into the embedding space used for memory storage and as prediction targets for the Embedding Predictor. It produces high-quality semantic embeddings optimized for being prediction targets in the JEPA training objective.

**Business Value:**
- Provides stable target embeddings for predictor training
- Enables semantic clustering in MCS
- Supports multiple encoder backends for quality/speed tradeoffs
- Separates query understanding (X-Encoder) from content representation (Y-Encoder)

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Embedding Quality | >0.85 | Semantic textual similarity benchmark |
| Encoding Latency | <50ms | p50 for 512-token input |
| Training Stability | <5% drift | Y-Encoder embedding variance during predictor training |

### Target Users

- MCS (content embedding for storage)
- Predictor training pipeline (target embeddings)
- Content similarity analysis tools

---

## 2. Functional Requirements

### Core Capabilities

- **FR-ENC-001**: The system shall encode text content to 1024-dimensional embeddings.
- **FR-ENC-002**: The system shall L2-normalize output embeddings (unit vectors).
- **FR-ENC-003**: The system shall support batch encoding for efficiency.
- **FR-ENC-004**: The system shall truncate inputs exceeding max context length (512 tokens).
- **FR-ENC-005**: The system shall support multiple backend models (configurable).
- **FR-ENC-006**: The system shall cache embeddings for repeated content (optional).
- **FR-ENC-007**: The system shall expose model info (name, dimension, max length).

### User Stories

- **US-ENC-001**: As the MCS, I want to encode content so that I can store and cluster memories.
- **US-ENC-002**: As the training pipeline, I want target embeddings so that I can train the predictor.
- **US-ENC-003**: As a developer, I want to switch encoder models so that I can optimize for quality or speed.
- **US-ENC-004**: As the ingestion pipeline, I want to batch-encode documents so that I can efficiently populate memory.

### Business Rules and Constraints

- **BR-ENC-001**: Output embeddings MUST be L2-normalized.
- **BR-ENC-002**: Output dimension MUST be consistent with configured model.
- **BR-ENC-003**: Y-Encoder MUST use slower learning rate (0.05×) during predictor training.
- **BR-ENC-004**: Y-Encoder weights SHOULD be EMA-updated during training (not direct gradient).

---

## 3. Non-Functional Requirements

### Performance Targets

| Metric | Target | Constraint |
|--------|--------|------------|
| Single Encoding Latency | <50ms p50 | <100ms p95 |
| Batch Encoding (32) | <500ms | <1s p95 |
| Memory Footprint | <2GB | Model + cache |
| GPU Memory | <4GB | Inference on RTX 3080 |

### Security Requirements

- **SEC-ENC-001**: Model weights MUST be loaded from trusted sources only.
- **SEC-ENC-002**: Input content MUST NOT be logged without explicit consent.

### Scalability Considerations

- Stateless encoding (no conversation state)
- Horizontal scaling via replicas
- GPU acceleration with batching

---

## 4. Features & Flows

### Feature Breakdown

| Feature | Priority | Description |
|---------|----------|-------------|
| Single Encoding | P0 | Encode single text to embedding |
| Batch Encoding | P0 | Encode multiple texts efficiently |
| Model Loading | P0 | Load pretrained encoder models |
| Model Switching | P1 | Runtime model configuration |
| Embedding Cache | P2 | LRU cache for repeated content |
| ONNX Export | P2 | Export for optimized inference |

### Key User Flows

**Flow 1: Encode Content**

```
Input: Text content string
  │
  ▼
┌─────────────────┐
│ Tokenize        │ ──▶ Token IDs + Attention Mask
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Truncate        │ ──▶ Max 512 tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Base Encoder    │ ──▶ Hidden states
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mean Pooling    │ ──▶ Pooled embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ L2 Normalize    │ ──▶ Unit vector
└─────────────────┘

Output: Target embedding S_Y (1024-dim)
```

### Input/Output Specifications

**Encode Input:**
```python
class EncodeInput(BaseModel):
    """Input for Y-Encoder."""

    content: str = Field(min_length=1, max_length=50000)


class BatchEncodeInput(BaseModel):
    """Input for batch Y-Encoder."""

    contents: list[str] = Field(min_length=1, max_length=1000)
```

**Encode Output:**
```python
class EncodeOutput(BaseModel):
    """Output from Y-Encoder."""

    embedding: list[float] = Field(description="Target embedding (D)")
    token_count: int = Field(ge=0, description="Number of tokens")
    truncated: bool = Field(description="True if input was truncated")


class BatchEncodeOutput(BaseModel):
    """Output from batch Y-Encoder."""

    embeddings: list[list[float]]
    token_counts: list[int]
    truncated: list[bool]
```

---

## 5. Code Pattern Requirements

### Naming Conventions

- **Classes**: PascalCase (`YEncoder`, `EncoderConfig`)
- **Functions**: snake_case (`encode`, `encode_batch`, `load_model`)
- **Variables**: snake_case (`embedding_dim`, `max_length`)
- **Constants**: SCREAMING_SNAKE_CASE (`DEFAULT_MODEL_NAME`)

### Type Safety Requirements

- **Type hint coverage**: 100%
- **Tensor types**: `torch.Tensor` for internal, `list[float]` for API
- **Required import**: `from __future__ import annotations`

### Testing Approach

- **Framework**: pytest
- **Coverage requirement**: ≥90%

**Required Test Cases:**
- `test_encode_returns_correct_dim`
- `test_output_normalized`
- `test_batch_encode_consistency`
- `test_truncation_long_input`
- `test_model_loading`
- `test_different_backends`

### Error Handling

- **Strategy**: Explicit raises
- **Custom exceptions**: `EncoderError`, `ModelLoadError`
- **Validation**: Input validation at encode() entry

### Architecture Patterns

- **Module structure**: `src/prime/encoder/y_encoder.py`
- **Protocol**: Implement `Encoder` protocol for interchangeability with X-Encoder

---

## 6. Acceptance Criteria

### Definition of Done

- [ ] Encode produces 1024-dim L2-normalized embeddings
- [ ] Batch encoding works correctly
- [ ] All supported models load and encode
- [ ] Latency targets met
- [ ] Unit tests passing with ≥90% coverage
- [ ] Type checking passes

### Validation Approach

1. **Unit Testing**: pytest with synthetic inputs
2. **Integration Testing**: Y-Encoder + MCS write path
3. **Quality Testing**: STS benchmark evaluation

---

## 7. Dependencies

### Technical Assumptions

- HuggingFace Transformers for model loading
- PyTorch for inference
- GPU available for acceleration

### External Integrations

| Integration | Type | Purpose |
|-------------|------|---------|
| Transformers | Required | Model loading and inference |
| PyTorch | Required | Tensor operations |
| Sentence-Transformers | Optional | Pre-trained encoders |

### Related Components

| Component | Relationship |
|-----------|--------------|
| MCS | Consumer (content embedding) |
| Predictor | Training target provider |

---

## 8. Configuration Schema

```python
class YEncoderConfig(BaseModel):
    """Configuration for Y-Encoder."""

    model_name: str = Field(
        default="google/gemma-embedding-300m",
        description="HuggingFace model name or path"
    )
    embedding_dim: int = Field(
        default=1024,
        description="Output embedding dimension"
    )
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum input tokens"
    )
    pooling_mode: str = Field(
        default="mean",
        description="Pooling strategy: 'mean', 'cls', 'max'"
    )
    normalize: bool = Field(
        default=True,
        description="L2 normalize output embeddings"
    )
    device: str = Field(
        default="cuda",
        description="Device for inference: 'cuda', 'cpu', 'mps'"
    )
    cache_size: int = Field(
        default=0,
        ge=0,
        description="LRU cache size (0 = disabled)"
    )

    model_config = {"frozen": True}
```

---

## 9. Supported Models

| Model | Parameters | Dimension | Strengths | Use Case |
|-------|------------|-----------|-----------|----------|
| `google/gemma-embedding-300m` | 300M | 1024 | Default, balanced | General purpose |
| `all-MiniLM-L6-v2` | 22M | 384 | Fast, lightweight | Low-latency |
| `BAAI/bge-large-en-v1.5` | 335M | 1024 | High quality | Quality-critical |
| `Qwen3-Embedding-0.6B` | 600M | 1024 | Multilingual | International |
| `Qwen3-Embedding-8B` | 8B | 1024 | Highest quality | Premium tier |

---

## 10. Protocol Definition

```python
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """Protocol for embedding encoders (X and Y)."""

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...

    def encode(self, text: str) -> np.ndarray:
        """Encode single text to embedding vector."""
        ...

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode batch of texts."""
        ...
```

---

## Appendix: Source Traceability

| Requirement | Source Document | Section |
|-------------|-----------------|---------|
| FR-ENC-001 | PRIME-Project-Overview.md | 4.4.2 Architecture |
| FR-ENC-003 | PRIME-Project-Overview.md | 4.4.3 Training Configuration |
| Model Options | PRIME-Project-Overview.md | 4.4.4 Alternative Y-Encoders |
| Multimodal | enhancement.md | Multimodal Capabilities |
