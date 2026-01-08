# PRIME REST API Guide

This guide demonstrates how to use the PRIME REST API for predictive retrieval.

## Starting the Server

### Basic Start

```bash
# Using uvicorn directly
uvicorn prime.api.app:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
uvicorn prime.api.app:app --reload

# Using uv
uv run uvicorn prime.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Configuration via Environment

```bash
# Configure via environment variables
export PRIME_HOST=0.0.0.0
export PRIME_PORT=8000
export PRIME_WORKERS=4

# For Qdrant (production)
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_API_KEY=your-api-key

# Start server
uvicorn prime.api.app:app
```

### Production Deployment

```bash
# Multiple workers (production)
uvicorn prime.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn
gunicorn prime.api.app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## API Endpoints

### Health Check

Check if the server is running:

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Process Turn

Process a conversation turn with predictive retrieval. This is the main endpoint.

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is machine learning?",
    "session_id": "demo_session",
    "force_retrieval": false,
    "k": 5
  }'
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | Yes | User input text |
| `session_id` | string | No | Session ID for stateful conversations (auto-generated if omitted) |
| `force_retrieval` | bool | No | Force retrieval regardless of SSM decision (default: false) |
| `k` | int | No | Number of memories to retrieve (default: 5) |
| `user_id` | string | No | User identifier for multi-tenant scenarios |

**Response:**

```json
{
  "retrieved_memories": [
    {
      "memory_id": "mem_abc123",
      "content": "Machine learning is a subset of AI...",
      "cluster_id": 0,
      "similarity": 0.89,
      "metadata": {"source": "external"},
      "created_at": 1704123456.789
    }
  ],
  "boundary_crossed": true,
  "variance": 0.18,
  "smoothed_variance": 0.15,
  "action": "retrieve",
  "session_id": "demo_session",
  "turn_number": 1,
  "latency_ms": 45.2
}
```

**Action Values:**

| Action | Meaning |
|--------|---------|
| `continue` | Stay on topic, no retrieval needed |
| `prepare` | Approaching boundary, caches pre-warmed |
| `retrieve` | Topic shift detected, context retrieved |
| `retrieve_consolidate` | Major shift, retrieved + consolidated clusters |

---

### Write Memory

Store content in memory for future retrieval:

```bash
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{
    "content": "JEPA predicts in embedding space rather than pixel space.",
    "metadata": {
      "source": "research_paper",
      "topic": "ml_architectures"
    },
    "session_id": "demo_session"
  }'
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | Content to store |
| `metadata` | object | No | Key-value metadata for filtering |
| `session_id` | string | No | Session identifier |
| `user_id` | string | No | User identifier |

**Response:**

```json
{
  "memory_id": "mem_xyz789",
  "cluster_id": 2,
  "is_new_cluster": false,
  "consolidated": false
}
```

---

### Search Memory

Direct memory search (bypasses SSM boundary detection):

```bash
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural network architectures",
    "k": 5,
    "min_similarity": 0.5
  }'
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query text |
| `k` | int | No | Number of results (default: 5) |
| `min_similarity` | float | No | Minimum similarity threshold (default: 0.0) |
| `session_id` | string | No | Filter by session |
| `user_id` | string | No | Filter by user |

**Response:**

```json
{
  "results": [
    {
      "memory_id": "mem_abc123",
      "content": "Transformer architecture uses self-attention...",
      "cluster_id": 1,
      "similarity": 0.92,
      "metadata": {"topic": "transformers"},
      "created_at": 1704123456.789
    }
  ],
  "query_embedding": null
}
```

---

### Get Diagnostics

System health and performance metrics:

```bash
curl http://localhost:8000/api/v1/diagnostics
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "components": {
    "ssm": {
      "name": "SemanticStateMonitor",
      "status": "healthy",
      "latency_p50_ms": 2.1,
      "error_rate": 0.0
    },
    "mcs": {
      "name": "MemoryClusterStore",
      "status": "healthy",
      "latency_p50_ms": 5.3,
      "error_rate": 0.0
    },
    "predictor": {
      "name": "EmbeddingPredictor",
      "status": "healthy",
      "latency_p50_ms": 12.8,
      "error_rate": 0.0
    },
    "y_encoder": {
      "name": "YEncoder",
      "status": "healthy",
      "latency_p50_ms": 8.5,
      "error_rate": 0.0
    }
  },
  "metrics": {
    "total_requests": 1523.0,
    "total_errors": 2.0,
    "error_rate": 0.0013,
    "active_sessions": 5.0,
    "ssm_turn_number": 42.0,
    "mcs_cluster_count": 15.0,
    "mcs_memory_count": 128.0
  }
}
```

---

### List Clusters

View memory cluster information:

```bash
curl "http://localhost:8000/api/v1/clusters?limit=10&offset=0"
```

**Response:**

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "size": 12,
      "is_consolidated": true,
      "representative_content": "Machine learning algorithms..."
    },
    {
      "cluster_id": 1,
      "size": 5,
      "is_consolidated": false,
      "representative_content": "Neural network training..."
    }
  ],
  "total": 15,
  "limit": 10,
  "offset": 0
}
```

---

### Get/Update Configuration

View current configuration:

```bash
curl http://localhost:8000/api/v1/config
```

Update configuration (partial update):

```bash
curl -X PUT http://localhost:8000/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{
    "ssm": {
      "variance_threshold": 0.20
    }
  }'
```

---

## Complete Example Session

Here's a full session demonstrating the predictive retrieval workflow:

```bash
# 1. Check server health
curl http://localhost:8000/api/v1/health

# 2. Ingest some knowledge
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{"content": "PRIME uses variance-based boundary detection to decide when retrieval is needed."}'

curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{"content": "The Embedding Predictor predicts target embeddings before searching."}'

curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -d '{"content": "Memory Cluster Store automatically consolidates similar memories."}'

# 3. Start a conversation
# First turn - establishes topic
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input": "What is PRIME?", "session_id": "demo"}'

# Second turn - same topic, probably no retrieval
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input": "How does the predictor work?", "session_id": "demo"}'

# Third turn - topic shift, should trigger retrieval
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the weather like?", "session_id": "demo"}'

# 4. Force retrieval for testing
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input": "Tell me about memory consolidation", "session_id": "demo", "force_retrieval": true}'

# 5. Direct search
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "boundary detection", "k": 3}'

# 6. Check diagnostics
curl http://localhost:8000/api/v1/diagnostics
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message here",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Status | Code | Meaning |
|--------|------|---------|
| 400 | `VALIDATION_ERROR` | Invalid request body |
| 401 | `AUTHENTICATION_ERROR` | Missing/invalid API key |
| 404 | `NOT_FOUND` | Resource not found |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server error |

### Rate Limiting

The API includes rate limiting (default: 60 requests/minute). When exceeded:

```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 42
}
```

Check the `Retry-After` header for wait time in seconds.

---

## Authentication (Optional)

If API key authentication is enabled:

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"input": "Hello"}'
```

Configure via environment:

```bash
export PRIME_API_KEY=your-secret-key
```

---

## OpenAPI Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
