# PRIME API Specification

## 1. Overview

### Purpose and Business Value

The PRIME API provides the unified interface for the PRIME system, including the main orchestration class (`PRIME`) and FastAPI REST endpoints. It integrates all core components (SSM, Predictor, MCS, Y-Encoder) into a cohesive system that applications can use for intelligent retrieval-augmented generation.

**Business Value:**
- Single entry point for all PRIME functionality
- REST API for language-agnostic integration
- LangChain/LlamaIndex adapters for ecosystem compatibility
- OpenAPI documentation for API consumers

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-End Latency | <200ms p50 | Full process_turn() |
| API Throughput | >100 QPS | Concurrent requests |
| Error Rate | <0.1% | Production monitoring |
| API Uptime | 99.9% | SLA target |

### Target Users

- Application developers (REST API)
- LangChain/LlamaIndex users (adapters)
- Internal components (PRIME class)

---

## 2. Functional Requirements

### PRIME Class

- **FR-API-001**: The system shall process conversation turns through the full pipeline.
- **FR-API-002**: The system shall record LLM responses to memory.
- **FR-API-003**: The system shall allow writing external knowledge to memory.
- **FR-API-004**: The system shall provide system diagnostics.
- **FR-API-005**: The system shall support force_retrieval override.
- **FR-API-006**: The system shall support configuration updates at runtime.

### REST API

- **FR-API-007**: The system shall expose POST /process endpoint for conversation turns.
- **FR-API-008**: The system shall expose POST /memory/write for external knowledge.
- **FR-API-009**: The system shall expose POST /memory/search for direct search.
- **FR-API-010**: The system shall expose GET /diagnostics for system health.
- **FR-API-011**: The system shall expose GET /clusters for memory cluster listing.
- **FR-API-012**: The system shall expose PUT /config for configuration updates.
- **FR-API-013**: The system shall provide OpenAPI documentation at /docs.

### Adapters

- **FR-API-014**: The system shall provide LangChain BaseRetriever implementation.
- **FR-API-015**: The system shall provide LlamaIndex retriever implementation.

### Evaluation (RAGAS)

- **FR-API-016**: The system shall compute Faithfulness score for generated responses.
- **FR-API-017**: The system shall compute Context Precision score for retrieved contexts.
- **FR-API-018**: The system shall compute Answer Relevancy score for responses.
- **FR-API-019**: The system shall provide evaluation metrics via GET /diagnostics/eval endpoint.
- **FR-API-020**: The system shall support batch evaluation for quality monitoring.

### User Stories

- **US-API-001**: As an application developer, I want to send messages to PRIME and receive augmented context.
- **US-API-002**: As a LangChain user, I want a drop-in retriever so I can use PRIME in my chains.
- **US-API-003**: As an operator, I want health endpoints so I can monitor PRIME.
- **US-API-004**: As a knowledge admin, I want to ingest documents into PRIME memory.
- **US-API-005**: As an ML engineer, I want to evaluate retrieval quality so I can monitor and improve PRIME.

### Business Rules and Constraints

- **BR-API-001**: process_turn() MUST return within 500ms p99.
- **BR-API-002**: API MUST use JSON request/response format.
- **BR-API-003**: API MUST validate all inputs before processing.
- **BR-API-004**: API MUST return structured error responses with error codes.

---

## 3. Non-Functional Requirements

### Performance Targets

| Metric | Target | Constraint |
|--------|--------|------------|
| process_turn Latency | <200ms p50 | <500ms p99 |
| Memory Write Latency | <50ms p50 | <100ms p95 |
| Memory Search Latency | <80ms p50 | <150ms p95 |
| Concurrent Connections | 1000 | Per instance |
| Request Body Size | <10MB | Max limit |

### Security Requirements

- **SEC-API-001**: API MUST support authentication (API key or JWT).
- **SEC-API-002**: API MUST rate limit by client identity.
- **SEC-API-003**: API MUST validate Content-Type headers.
- **SEC-API-004**: API MUST sanitize inputs to prevent injection.

### Scalability Considerations

- Stateless API servers behind load balancer
- Session state in SSM requires sticky sessions OR state externalization
- Horizontal scaling via replicas

---

## 4. Features & Flows

### Feature Breakdown

| Feature | Priority | Description |
|---------|----------|-------------|
| process_turn | P0 | Main conversation processing |
| memory/write | P0 | External knowledge ingestion |
| memory/search | P0 | Direct memory search |
| diagnostics | P1 | System health and metrics |
| clusters | P1 | Memory cluster inspection |
| config | P2 | Runtime configuration |
| LangChain Adapter | P2 | Framework integration |
| LlamaIndex Adapter | P2 | Framework integration |
| **RAGAS Evaluation** | P1 | Reference-free RAG quality metrics |

### Key User Flows

**Flow 1: Process Conversation Turn**

```
POST /process
{
  "input": "What is the capital of France?",
  "session_id": "sess_123",
  "force_retrieval": false
}
  │
  ▼
┌─────────────────┐
│ Input Validation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SSM Update      │ ──▶ Check boundary
└────────┬────────┘
         │
    ┌────┴────┐
   No        Yes (boundary crossed)
    │         │
    │         ▼
    │   ┌─────────────┐
    │   │ Predictor   │ ──▶ Generate Ŝ_Y
    │   └──────┬──────┘
    │          │
    │          ▼
    │   ┌─────────────┐
    │   │ MCS Search  │ ──▶ Retrieved memories
    │   └──────┬──────┘
    │          │
    └────┬─────┘
         │
         ▼
┌─────────────────┐
│ Format Response │
└─────────────────┘

Response:
{
  "retrieved_memories": [...],
  "boundary_crossed": true,
  "variance": 0.23,
  "action": "retrieve",
  "session_id": "sess_123"
}
```

### API Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/process` | POST | Process conversation turn | Required |
| `/memory/write` | POST | Write external knowledge | Required |
| `/memory/search` | POST | Direct memory search | Required |
| `/diagnostics` | GET | System diagnostics | Optional |
| `/clusters` | GET | List memory clusters | Required |
| `/clusters/{id}` | GET | Get cluster details | Required |
| `/config` | PUT | Update configuration | Admin |
| `/health` | GET | Health check | None |
| `/docs` | GET | OpenAPI documentation | None |
| `/diagnostics/eval` | POST | RAGAS evaluation metrics | Admin |

### Input/Output Specifications

**POST /process**

Request:
```python
class ProcessRequest(BaseModel):
    """Request for processing a conversation turn."""

    input: str = Field(min_length=1, max_length=8192)
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str | None = None
    force_retrieval: bool = False
    k: int = Field(default=5, ge=1, le=20)
```

Response:
```python
class ProcessResponse(BaseModel):
    """Response from process_turn."""

    retrieved_memories: list[MemoryReadResult]
    boundary_crossed: bool
    variance: float
    smoothed_variance: float
    action: ActionState
    session_id: str
    turn_number: int
    latency_ms: float
```

**POST /memory/write**

Request:
```python
class MemoryWriteRequest(BaseModel):
    """Request to write memory."""

    content: str = Field(min_length=1, max_length=50000)
    metadata: dict[str, str | int | float | bool] | None = None
    user_id: str | None = None
    session_id: str | None = None
```

Response:
```python
class MemoryWriteResponse(BaseModel):
    """Response from memory write."""

    memory_id: str
    cluster_id: int
    is_new_cluster: bool
    consolidated: bool
```

**POST /memory/search**

Request:
```python
class MemorySearchRequest(BaseModel):
    """Request for direct memory search."""

    query: str = Field(min_length=1, max_length=8192)
    k: int = Field(default=5, ge=1, le=100)
    user_id: str | None = None
    session_id: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
```

Response:
```python
class MemorySearchResponse(BaseModel):
    """Response from memory search."""

    results: list[MemoryReadResult]
    query_embedding: list[float] | None = None
```

**GET /diagnostics**

Response:
```python
class DiagnosticsResponse(BaseModel):
    """System diagnostics response."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    components: dict[str, ComponentStatus]
    metrics: dict[str, float]


class ComponentStatus(BaseModel):
    """Status of individual component."""

    name: str
    status: str
    latency_p50_ms: float
    error_rate: float
```

**POST /diagnostics/eval**

Request:
```python
class EvaluationRequest(BaseModel):
    """Request for RAGAS evaluation."""

    query: str = Field(min_length=1, max_length=8192)
    retrieved_contexts: list[str] = Field(min_items=1, max_items=20)
    generated_response: str = Field(min_length=1, max_length=50000)
    ground_truth: str | None = Field(default=None, description="Optional ground truth for reference-based metrics")
```

Response:
```python
class EvaluationResponse(BaseModel):
    """Response with RAGAS evaluation metrics."""

    faithfulness: float = Field(ge=0.0, le=1.0, description="How factually consistent is the response with retrieved contexts")
    context_precision: float = Field(ge=0.0, le=1.0, description="How relevant are the retrieved contexts to the query")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="How relevant is the response to the query")
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0, description="Context recall (requires ground_truth)")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted average of all metrics")
    latency_ms: float
```

---

## 5. Code Pattern Requirements

### Naming Conventions

- **Classes**: PascalCase (`PRIME`, `ProcessRequest`)
- **Functions**: snake_case (`process_turn`, `write_memory`)
- **Variables**: snake_case
- **API Routes**: kebab-case (`/memory/write`, `/clusters`)

### Type Safety Requirements

- **Type hint coverage**: 100%
- **Pydantic models**: All request/response bodies
- **Required import**: `from __future__ import annotations`

### Testing Approach

- **Framework**: pytest + httpx (async client)
- **Coverage requirement**: ≥85%

**Required Test Cases:**
- `test_process_turn_no_retrieval`
- `test_process_turn_with_retrieval`
- `test_memory_write`
- `test_memory_search`
- `test_diagnostics`
- `test_authentication_required`
- `test_rate_limiting`
- `test_input_validation`
- `test_ragas_evaluation`
- `test_faithfulness_score`
- `test_context_precision`

### Error Handling

- **Strategy**: Structured error responses
- **HTTP status codes**: 400 (validation), 401 (auth), 429 (rate limit), 500 (server)
- **Error response format**:
```python
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    error_code: str
    detail: str | None = None
    request_id: str
```

### Architecture Patterns

- **Module structure**:
  - `src/prime/prime.py` - Main PRIME class
  - `src/prime/api/app.py` - FastAPI application
  - `src/prime/api/routes/` - Route modules
  - `src/prime/api/middleware/` - Auth, rate limiting
  - `src/prime/adapters/` - LangChain, LlamaIndex

---

## 6. Acceptance Criteria

### Definition of Done

- [ ] All endpoints implemented and documented
- [ ] OpenAPI spec generated at /docs
- [ ] Authentication middleware working
- [ ] Rate limiting implemented
- [ ] End-to-end latency <200ms p50
- [ ] Unit tests passing with ≥85% coverage
- [ ] Integration tests with all components
- [ ] Docker containerization complete
- [ ] Type checking passes

### Validation Approach

1. **Unit Testing**: pytest with mocked components
2. **Integration Testing**: Full stack with test database
3. **Load Testing**: Locust or k6 for throughput
4. **Contract Testing**: OpenAPI spec validation

---

## 7. Dependencies

### Technical Assumptions

- All core components (SSM, MCS, Predictor, Y-Encoder) available
- Redis available for rate limiting and caching
- PostgreSQL for session state (optional)

### External Integrations

| Integration | Type | Purpose |
|-------------|------|---------|
| FastAPI | Required | REST framework |
| Uvicorn | Required | ASGI server |
| Pydantic | Required | Data validation |
| Redis | Optional | Rate limiting, caching |
| LangChain | Optional | Adapter support |

### Related Components

| Component | Relationship |
|-----------|--------------|
| SSM | Internal dependency |
| MCS | Internal dependency |
| Predictor | Internal dependency |
| Y-Encoder | Internal dependency |

---

## 8. Configuration Schema

```python
class PRIMEConfig(BaseModel):
    """Main PRIME system configuration."""

    # Component configs
    ssm: SSMConfig = Field(default_factory=SSMConfig)
    mcs: MCSConfig = Field(default_factory=MCSConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    y_encoder: YEncoderConfig = Field(default_factory=YEncoderConfig)

    # API config
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=60)

    # LLM integration
    llm_provider: str = Field(default="anthropic")
    llm_model: str = Field(default="claude-3.5-sonnet")

    # RAGAS Evaluation Configuration
    enable_ragas_eval: bool = Field(
        default=True,
        description="Enable RAGAS evaluation metrics"
    )
    ragas_llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM for RAGAS evaluation (requires reasoning)"
    )
    ragas_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for evaluation"
    )

    model_config = {"frozen": True}
```

---

## 9. PRIME Class Implementation

```python
class PRIME:
    """Main PRIME orchestration class."""

    def __init__(self, config: PRIMEConfig) -> None:
        self.config = config
        self.ssm = SemanticStateMonitor(config.ssm)
        self.mcs = MemoryClusterStore(config.mcs)
        self.predictor = EmbeddingPredictor(config.predictor)
        self.y_encoder = YEncoder(config.y_encoder)

    def process_turn(
        self,
        input_text: str,
        session_id: str,
        force_retrieval: bool = False,
        k: int = 5,
    ) -> PRIMEResponse:
        """Process a conversation turn through the PRIME pipeline.

        Args:
            input_text: User input text.
            session_id: Session identifier for SSM state.
            force_retrieval: If True, bypass SSM and force retrieval.
            k: Number of memories to retrieve.

        Returns:
            PRIMEResponse with retrieved memories and diagnostics.

        Raises:
            PRIMEError: If processing fails.
        """
        ...

    def record_response(
        self,
        response: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Record LLM response to memory.

        Args:
            response: LLM response text.
            session_id: Session identifier.
            metadata: Optional metadata.

        Returns:
            MemoryWriteResult with cluster assignment.
        """
        ...

    def write_external_knowledge(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Write external knowledge to memory.

        Args:
            content: Knowledge content.
            metadata: Optional metadata.

        Returns:
            MemoryWriteResult with cluster assignment.
        """
        ...

    def get_diagnostics(self) -> PRIMEDiagnostics:
        """Get system diagnostics.

        Returns:
            PRIMEDiagnostics with component health and metrics.
        """
        ...
```

---

## 10. LangChain Adapter

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class PRIMERetriever(BaseRetriever):
    """PRIME as a LangChain retriever.

    Example:
        ```python
        from prime.adapters.langchain import PRIMERetriever

        retriever = PRIMERetriever(
            prime_url="http://localhost:8000",
            api_key="your-api-key",
        )

        docs = retriever.invoke("What is PRIME?")
        ```
    """

    prime_client: PRIMEClient
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents from PRIME.

        Args:
            query: Query text.
            run_manager: Optional callback manager.

        Returns:
            List of LangChain Documents.
        """
        result = self.prime_client.search(query=query, k=self.k)

        return [
            Document(
                page_content=memory.content,
                metadata={
                    "memory_id": memory.memory_id,
                    "cluster_id": memory.cluster_id,
                    "similarity": memory.similarity,
                    **memory.metadata,
                },
            )
            for memory in result.results
        ]
```

---

## 11. RAGAS Evaluation Implementation

RAGAS (Reference-free Automatic evaluation of Generated Answers for Search) provides reference-free metrics for RAG quality assessment.

### Core Metrics

| Metric | Description | LLM Required |
|--------|-------------|--------------|
| Faithfulness | Factual consistency between response and contexts | Yes |
| Context Precision | Relevance of retrieved contexts to query | Yes |
| Answer Relevancy | Relevance of response to original query | Yes |
| Context Recall | Coverage of ground truth (requires reference) | Yes |

### Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    answer_relevancy,
    context_recall,
)
from datasets import Dataset


class RAGASEvaluator:
    """RAGAS evaluation service for PRIME."""

    def __init__(self, config: PRIMEConfig) -> None:
        self.config = config
        self.metrics = [
            faithfulness,
            context_precision,
            answer_relevancy,
        ]
        if config.enable_ragas_eval:
            self._setup_llm()

    def _setup_llm(self) -> None:
        """Configure LLM for RAGAS evaluation."""
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=self.config.ragas_llm_model,
            temperature=0,
        )
        # Set LLM for all metrics
        for metric in self.metrics:
            metric.llm = self.llm

    def evaluate(
        self,
        query: str,
        retrieved_contexts: list[str],
        generated_response: str,
        ground_truth: str | None = None,
    ) -> EvaluationResponse:
        """Evaluate a single RAG response.

        Args:
            query: User query.
            retrieved_contexts: List of retrieved context strings.
            generated_response: LLM-generated response.
            ground_truth: Optional ground truth for context_recall.

        Returns:
            EvaluationResponse with all computed metrics.
        """
        import time
        start = time.perf_counter()

        data = {
            "question": [query],
            "contexts": [retrieved_contexts],
            "answer": [generated_response],
        }

        metrics = self.metrics.copy()
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)

        latency_ms = (time.perf_counter() - start) * 1000

        scores = {
            "faithfulness": result["faithfulness"],
            "context_precision": result["context_precision"],
            "answer_relevancy": result["answer_relevancy"],
        }

        context_recall_score = None
        if ground_truth:
            context_recall_score = result["context_recall"]
            scores["context_recall"] = context_recall_score

        overall = sum(scores.values()) / len(scores)

        return EvaluationResponse(
            faithfulness=scores["faithfulness"],
            context_precision=scores["context_precision"],
            answer_relevancy=scores["answer_relevancy"],
            context_recall=context_recall_score,
            overall_score=overall,
            latency_ms=latency_ms,
        )

    async def evaluate_batch(
        self,
        samples: list[dict],
    ) -> list[EvaluationResponse]:
        """Evaluate multiple samples for quality monitoring.

        Args:
            samples: List of dicts with query, contexts, response keys.

        Returns:
            List of EvaluationResponse objects.
        """
        results = []
        for sample in samples:
            result = self.evaluate(
                query=sample["query"],
                retrieved_contexts=sample["contexts"],
                generated_response=sample["response"],
                ground_truth=sample.get("ground_truth"),
            )
            results.append(result)
        return results
```

### Quality Monitoring Integration

```python
@router.post("/diagnostics/eval")
async def evaluate_response(
    request: EvaluationRequest,
    evaluator: RAGASEvaluator = Depends(get_evaluator),
) -> EvaluationResponse:
    """Evaluate RAG response quality using RAGAS metrics.

    Returns:
        EvaluationResponse with faithfulness, context_precision,
        answer_relevancy, and optionally context_recall scores.
    """
    return evaluator.evaluate(
        query=request.query,
        retrieved_contexts=request.retrieved_contexts,
        generated_response=request.generated_response,
        ground_truth=request.ground_truth,
    )
```

---

## Appendix: Source Traceability

| Requirement | Source Document | Section |
|-------------|-----------------|---------|
| FR-API-001 | PRIME-Project-Overview.md | 5.1 Complete Request Flow |
| FR-API-007 | PRIME-Project-Overview.md | 5.3 API Endpoints |
| FR-API-014 | enhancement.md | LangChain/LlamaIndex Adapters |
| Performance | PRIME-Project-Overview.md | 9.2 Quantitative Targets |
| Security | strategic-intel.md | Enterprise Requirements |
| FR-API-016-020 | enhancement.md | RAGAS Evaluation Framework |
