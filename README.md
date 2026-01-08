# PRIME

**Predictive Retrieval with Intelligent Memory Embeddings**

_Predict what you need before you search._

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## What Makes PRIME Different

Traditional RAG systems are **reactive**: they embed your query and search for similar documents every single turn. This leads to:

- **Over-retrieval**: Wasted compute searching when context hasn't changed
- **Suboptimal results**: Query embeddings don't always match ideal context embeddings
- **Memory fragmentation**: Similar content stored as separate, redundant entries

PRIME is **predictive**. Inspired by Meta FAIR's [VL-JEPA](https://ai.meta.com/research/publications/vl-jepa-video-language-models-with-joint-embeddings/), PRIME:

| Problem               | PRIME's Solution                                                 |
| --------------------- | ---------------------------------------------------------------- |
| **When** to retrieve? | Semantic State Monitor detects topic shifts via variance         |
| **What** to retrieve? | Embedding Predictor predicts ideal context _before_ searching    |
| **How** to store?     | Memory Cluster Store consolidates similar memories automatically |

```
Traditional RAG:  Query → Embed → Search Every Time → Retrieve → Generate
        PRIME:    Query → Monitor Variance → Predict Target → Targeted Search → Generate
                           ↓                        ↓
                    (skip if same topic)    (search for predicted ideal, not query)
```

---

## Quick Start

### Installation

```bash
# Basic install
pip install prime-rag

# With all optional dependencies
pip install prime-rag[all]

# Development install from source
git clone https://github.com/Mathews-Tom/PRIME.git
cd PRIME
uv sync
```

### Library Usage (Recommended for Getting Started)

```python
from prime import PRIME, PRIMEConfig

# Initialize with testing config (in-memory, no external services)
config = PRIMEConfig.for_testing()
prime = PRIME(config)

# Process a conversation turn
response = prime.process_turn(
    "What is machine learning?",
    session_id="demo_session",
)

print(f"Action: {response.action.value}")  # continue, prepare, retrieve, or retrieve_consolidate
print(f"Boundary crossed: {response.boundary_crossed}")
print(f"Variance: {response.variance:.4f}")

# If retrieval was triggered, you get memories
if response.retrieved_memories:
    for mem in response.retrieved_memories:
        print(f"  [{mem.similarity:.2f}] {mem.content[:100]}...")

# Record a response to memory for future retrieval
prime.record_response(
    "Machine learning is a subset of AI that enables systems to learn from data...",
    session_id="demo_session",
)

# Ingest external knowledge
prime.write_external_knowledge(
    "JEPA (Joint Embedding Predictive Architecture) predicts in embedding space...",
    metadata={"source": "research_paper", "topic": "ml_architectures"},
)

# Direct memory search (bypasses SSM boundary detection)
results = prime.search_memory("neural network architectures", k=5)
```

### REST API Usage

Start the server:

```bash
# Using uvicorn directly
uvicorn prime.api.app:app --host 0.0.0.0 --port 8000

# Or with uv
uv run uvicorn prime.api.app:app --reload
```

Make requests:

```bash
# Process a turn
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input": "What is JEPA?", "session_id": "demo"}'

# Write to memory
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{"content": "JEPA predicts in embedding space rather than pixel space."}'

# Search memory
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "embedding prediction", "k": 5}'

# Health check
curl http://localhost:8000/api/v1/health
```

### Python SDK (for API Clients)

```python
from prime.client import PRIMEClient, PRIMEClientSync

# Async client
async with PRIMEClient(base_url="http://localhost:8000") as client:
    response = await client.process_turn("What is JEPA?")
    print(f"Action: {response.action}")

# Sync client
with PRIMEClientSync(base_url="http://localhost:8000") as client:
    response = client.process_turn("What is JEPA?")
    await client.write_memory("JEPA uses joint embedding spaces.")
```

---

## Framework Integrations

### LangChain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from prime import PRIME, PRIMEConfig
from prime.adapters.langchain import PRIMERetriever

# Initialize PRIME
prime = PRIME(PRIMEConfig.for_testing())

# Create LangChain retriever
retriever = PRIMERetriever(
    prime=prime,
    mode="process_turn",  # or "search" for direct search
    session_id="langchain_session",
    top_k=5,
)

# Use in a chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa_chain.invoke("What is predictive retrieval?")
```

### LlamaIndex

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from prime import PRIME, PRIMEConfig
from prime.adapters.llamaindex import PRIMELlamaIndexRetriever

# Initialize PRIME
prime = PRIME(PRIMEConfig.for_testing())

# Create LlamaIndex retriever
retriever = PRIMELlamaIndexRetriever(
    prime=prime,
    mode="process_turn",
    session_id="llamaindex_session",
)

# Build query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=OpenAI(model="gpt-4"),
)
response = query_engine.query("Explain memory consolidation in PRIME")
```

---

## Architecture

```plaintext
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRIME SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Input                                                                │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────-───┐                                 │
│   │    SEMANTIC STATE MONITOR (SSM)       │                                 │
│   │    ─────────────────────────────      │                                 │
│   │    • Encodes input via Y-Encoder      │                                 │
│   │    • Maintains sliding window         │                                 │
│   │    • Calculates variance from centroid│                                 │
│   │    • Triggers on boundary crossing    │                                 │
│   └──────────────┬─────────────────────-──┘                                 │
│                  │                                                          │
│        ┌─-────────┴─────────┐                                               │
│        │  Boundary crossed? │                                               │
│        └───-──────┬─────────┘                                               │
│           No │         │ Yes                                                │
│              │         ▼                                                    │
│              │   ┌────────────────────────────────────-──┐                  │
│              │   │    EMBEDDING PREDICTOR                │                  │
│              │   │    ───────────────────                │                  │
│              │   │    • Takes context window + query     │                  │
│              │   │    • Transformer predicts target Ŝ_Y  │                  │
│              │   │    • Trained with InfoNCE loss        │                  │
│              │   └──────────────┬──────────────────────-─┘                  │
│              │                  │                                           │
│              │                  ▼                                           │
│              │   ┌─────────────────────────────────────-─┐                  │
│              │   │    MEMORY CLUSTER STORE (MCS)         │                  │
│              │   │    ──────────────────────────         │                  │
│              │   │    • FAISS/Qdrant vector search       │                  │
│              │   │    • Searches with PREDICTED embedding│                  │
│              │   │    • Auto-clusters similar memories   │                  │
│              │   │    • Consolidates into prototypes     │                  │
│              │   └──────────────┬─────────────────────-──┘                  │
│              │                  │                                           │
│              ▼                  ▼                                           │
│   ┌────────────────────────────────────────────────────────-──┐             │
│   │                      RESPONSE                             │             │
│   │    • retrieved_memories: List[MemoryReadResult]           │             │
│   │    • boundary_crossed: bool                               │             │
│   │    • variance: float                                      │             │
│   │    • action: CONTINUE | PREPARE | RETRIEVE | CONSOLIDATE  │             │
│   └────────────────────────────────────────────────────────-──┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component                  | Purpose                   | Key Behavior                                                   |
| -------------------------- | ------------------------- | -------------------------------------------------------------- |
| **Semantic State Monitor** | Decide _when_ to retrieve | Tracks conversation trajectory; triggers on variance threshold |
| **Embedding Predictor**    | Decide _what_ to retrieve | Predicts ideal context embedding before search                 |
| **Memory Cluster Store**   | Storage + retrieval       | Auto-clusters, consolidates, and searches memories             |
| **Y-Encoder**              | Text → embedding          | Encodes content for storage and prediction targets             |

### Action States

```plaintext
CONTINUE  →  Stay on topic, no retrieval needed
PREPARE   →  Approaching boundary (variance > 0.7θ), pre-warm caches
RETRIEVE  →  Topic shift detected (variance > θ), retrieve context
RETRIEVE_CONSOLIDATE  →  Major shift (variance > 2θ), retrieve + consolidate clusters
```

---

## Configuration

### Environment Variables

```bash
# API Server
PRIME_HOST=0.0.0.0
PRIME_PORT=8000
PRIME_WORKERS=4
PRIME_RATE_LIMIT=60

# Vector Database (for production)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key  # optional

# Model Configuration
PRIME_ENCODER_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Evaluation
PRIME_RAGAS_ENABLED=true
PRIME_RAGAS_MODEL=gpt-4.1-mini
```

### Programmatic Configuration

```python
from prime import PRIMEConfig
from prime.ssm import SSMConfig
from prime.mcs import MCSConfig
from prime.predictor import PredictorConfig
from prime.encoder import YEncoderConfig

# Full custom configuration
config = PRIMEConfig(
    ssm=SSMConfig(
        variance_threshold=0.15,  # Lower = more sensitive
        window_size=5,            # Conversation turns to track
        smoothing_factor=0.3,     # EMA smoothing
    ),
    mcs=MCSConfig(
        similarity_threshold=0.85,   # Cluster membership cutoff
        consolidation_threshold=5,   # Min size to consolidate
        index_type="faiss",          # or "qdrant" for production
    ),
    predictor=PredictorConfig(
        input_dim=384,
        hidden_dim=768,
        output_dim=384,
        num_layers=2,
        num_heads=4,
    ),
    y_encoder=YEncoderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    ),
)

prime = PRIME(config)
```

### Preset Configurations

```python
# Testing (in-memory, small models)
config = PRIMEConfig.for_testing()

# Production (from environment variables)
config = PRIMEConfig.from_env()

# Validate for production deployment
config.validate_for_production()  # Raises if misconfigured
```

---

## API Reference

### Core Methods

| Method                                               | Description                                           | Returns                  |
| ---------------------------------------------------- | ----------------------------------------------------- | ------------------------ |
| `process_turn(text, session_id, force_retrieval, k)` | Process conversation turn with SSM boundary detection | `PRIMEResponse`          |
| `record_response(content, session_id, metadata)`     | Store LLM response to memory                          | `MemoryWriteResult`      |
| `write_external_knowledge(content, metadata)`        | Ingest external documents                             | `MemoryWriteResult`      |
| `search_memory(query, k, min_similarity)`            | Direct memory search (bypasses SSM)                   | `List[MemoryReadResult]` |
| `get_diagnostics()`                                  | System health and metrics                             | `PRIMEDiagnostics`       |
| `reset_session(session_id)`                          | Clear session state                                   | `None`                   |

### REST Endpoints

| Endpoint                | Method  | Description                            |
| ----------------------- | ------- | -------------------------------------- |
| `/api/v1/process`       | POST    | Process turn with predictive retrieval |
| `/api/v1/memory/write`  | POST    | Write content to memory                |
| `/api/v1/memory/search` | POST    | Search memory directly                 |
| `/api/v1/health`        | GET     | Health check                           |
| `/api/v1/diagnostics`   | GET     | System diagnostics                     |
| `/api/v1/clusters`      | GET     | List memory clusters                   |
| `/api/v1/config`        | GET/PUT | View/update configuration              |

---

## Examples

See the [`examples/`](./examples) directory for runnable demonstrations:

- **[`library_quickstart.py`](./examples/library_quickstart.py)** - In-process library usage
- **[`sdk_client.py`](./examples/sdk_client.py)** - Python SDK client examples
- **[`api_server.md`](./examples/api_server.md)** - REST API usage guide

---

## Production Notes

### Vector Database Selection

| Mode                  | Backend           | Use Case                                    |
| --------------------- | ----------------- | ------------------------------------------- |
| `index_type="faiss"`  | FAISS (in-memory) | Development, testing, small deployments     |
| `index_type="qdrant"` | Qdrant            | Production, persistence, horizontal scaling |

### Known Limitations (Alpha)

1. **Cluster state is in-memory**: Even with Qdrant for vectors, cluster bookkeeping (membership, prototypes) lives in-process. A service restart loses cluster state unless you implement persistence.

2. **Single-process sessions**: Session context (for the Predictor) is stored in-memory per process. Multi-worker deployments need external session storage.

3. **Predictor is untrained by default**: The Embedding Predictor initializes with random weights. For production quality, train on your domain using the InfoNCE objective.

### Scaling Recommendations

```python
# For production with Qdrant
config = PRIMEConfig(
    mcs=MCSConfig(
        index_type="qdrant",
        qdrant_host="qdrant.your-infra.com",
        qdrant_port=6333,
        qdrant_api_key="...",
    ),
)
```

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Type checking
uv run mypy src

# Linting
uv run ruff check .
uv run ruff check . --fix  # Auto-fix
```

---

## Project Status

PRIME is in **alpha**. The core architecture is implemented and functional, but:

- The Embedding Predictor needs training on domain-specific data
- Production deployment patterns are still being refined
- API may change before 1.0

Contributions and feedback welcome.

---

## References

- [VL-JEPA: Video-Language Joint Embedding Predictive Architecture](https://ai.meta.com/research/publications/vl-jepa-video-language-models-with-joint-embeddings/) - The architectural inspiration
- [PRIME Project Overview](./docs/PRIME-Project-Overview.md) - Full design specification
- [InfoNCE Loss](https://arxiv.org/abs/1807.03748) - Training objective for the Predictor

---

## License

MIT License - see [LICENSE](./LICENSE) for details.

---

_PRIME: Because the best retrieval is the one you predicted you'd need._
