# Encoder Model Trade-offs

This document outlines the supported embedding models and their trade-offs to help choose the right model for your use case.

## Supported Models

| Model | ID | Dimension | Max Length | Speed | Quality | Memory |
|-------|-----|-----------|------------|-------|---------|--------|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | ⚡ Fast | ★★★ | ~90 MB |
| BGE Large | `BAAI/bge-large-en-v1.5` | 1024 | 512 | ⏱️ Medium | ★★★★★ | ~1.3 GB |
| Gemma Embedding | `google/gemma-embedding-300m` | 1024 | 512 | ⏱️ Medium | ★★★★ | ~600 MB |
| Qwen3 0.6B | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 8192 | ⏱️ Medium | ★★★★ | ~1.2 GB |
| Qwen3 8B | `Qwen/Qwen3-Embedding-8B` | 1024 | 8192 | 🐢 Slow | ★★★★★ | ~16 GB |

## Model Selection Guide

### Development & Testing: MiniLM

```python
from prime.encoder import MINILM_CONFIG, YEncoder

encoder = YEncoder(MINILM_CONFIG)
```

**Best for:**
- Rapid prototyping
- Unit tests
- CI/CD pipelines
- Resource-constrained environments

**Trade-offs:**
- Lower embedding dimension (384 vs 1024)
- Shorter max context (256 tokens)
- Good-but-not-best semantic quality

### Production Default: Gemma Embedding

```python
from prime.encoder import GEMMA_EMBEDDING_CONFIG, YEncoder

encoder = YEncoder(GEMMA_EMBEDDING_CONFIG)
```

**Best for:**
- General production workloads
- Balanced quality/speed
- Standard RAG applications

**Trade-offs:**
- Requires ~600 MB memory
- Medium inference speed
- English-focused (limited multilingual)

### High Quality: BGE Large

```python
from prime.encoder import BGE_LARGE_CONFIG, YEncoder

encoder = YEncoder(BGE_LARGE_CONFIG)
```

**Best for:**
- High-stakes retrieval
- Legal/medical documents
- When quality matters most

**Trade-offs:**
- Uses CLS pooling (not mean pooling)
- May benefit from instruction prefixes
- ~1.3 GB memory footprint

### Long Context & Multilingual: Qwen3

```python
from prime.encoder import QWEN_EMBEDDING_CONFIG, YEncoder

encoder = YEncoder(QWEN_EMBEDDING_CONFIG)
```

**Best for:**
- Long documents (up to 8192 tokens)
- Multilingual content
- Asian language support

**Trade-offs:**
- Requires `trust_remote_code=True`
- ~1.2 GB memory (0.6B model)
- Newer model, less battle-tested

## Performance Characteristics

### Embedding Speed (approx.)

| Model | CPU (ms/text) | GPU (ms/text) |
|-------|---------------|---------------|
| MiniLM | ~15 | ~2 |
| Gemma | ~50 | ~8 |
| BGE Large | ~80 | ~12 |
| Qwen3 0.6B | ~60 | ~10 |

### Semantic Quality (MTEB Benchmark)

| Model | Average Score |
|-------|---------------|
| MiniLM | 0.56 |
| BGE Large | 0.64 |
| Qwen3 0.6B | 0.62 |

## Custom Model Configuration

For models not in presets:

```python
from prime.encoder import YEncoder, YEncoderConfig

config = YEncoderConfig(
    model_name="your-org/custom-model",
    embedding_dim=768,  # Must match model's actual dimension
    max_length=512,
    pooling_mode="mean",  # or "cls", "max"
    normalize=True,
    device="auto",
    trust_remote_code=False,
)

encoder = YEncoder(config)
```

## Dimension Validation

PRIME validates that the configured `embedding_dim` matches the model's actual output dimension at load time. A `ModelLoadError` is raised if there's a mismatch:

```python
# This will raise ModelLoadError
config = YEncoderConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=1024,  # Wrong! MiniLM produces 384
)
encoder = YEncoder(config)  # Raises ModelLoadError
```

## Pooling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `mean` | Average of all token embeddings | General use, most models |
| `cls` | First token ([CLS]) embedding | BERT-style models, BGE |
| `max` | Max pooling over sequence | Rare, specific use cases |

Most sentence transformer models work best with `mean` pooling. BGE models are trained with `cls` pooling.
