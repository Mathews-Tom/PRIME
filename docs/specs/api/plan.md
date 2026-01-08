# API-001: PRIME API - Implementation Plan (PRP)

## Executive Summary

### Component Overview

The PRIME API serves as the unified interface layer for the PRIME system, orchestrating all core components (SSM, Predictor, MCS, Y-Encoder) into a cohesive REST API. This component provides the primary integration point for applications, framework adapters (LangChain/LlamaIndex), and quality evaluation via RAGAS metrics.

### Key Objectives

1. Implement `PRIME` orchestration class coordinating all components
2. Build FastAPI REST endpoints with authentication and rate limiting
3. Create LangChain and LlamaIndex adapters for ecosystem compatibility
4. Integrate RAGAS evaluation for reference-free RAG quality metrics
5. Achieve <200ms p50 latency for `process_turn` operations

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-End Latency | <200ms p50, <500ms p99 | `process_turn()` timing |
| API Throughput | >100 QPS | Concurrent requests |
| Error Rate | <0.1% | Production monitoring |
| Test Coverage | ≥85% | pytest-cov |

### Dependencies

| Component | Status | Relationship |
|-----------|--------|--------------|
| ENC-001 Y-Encoder | Required | Text to embedding conversion |
| SSM-001 Semantic State Monitor | Required | Retrieval decision |
| MCS-001 Memory Cluster Store | Required | Memory storage/retrieval |
| PRED-001 Embedding Predictor | Required | Target embedding prediction |
| FastAPI | External | REST framework |
| Redis | Optional | Rate limiting, caching |
| RAGAS | External | Evaluation metrics |

---

## Architecture Design

### System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              External Clients                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ HTTP Client  │  │   LangChain  │  │  LlamaIndex  │  │ Evaluation CLI   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
└─────────┼─────────────────┼─────────────────┼───────────────────┼───────────┘
          │                 │                 │                   │
          ▼                 ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRIME API Layer                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Application                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ /process    │  │ /memory/*   │  │ /diagnostics│  │ /health     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│  ┌───────────────────────────────▼──────────────────────────────────────┐   │
│  │                        Middleware Stack                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │    Auth     │  │ Rate Limit  │  │  Logging    │  │   CORS      │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│  ┌───────────────────────────────▼──────────────────────────────────────┐   │
│  │                         PRIME Class                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │ process_turn() | record_response() | write_external_knowledge() │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│  ┌──────────────┬────────────────┼────────────────┬──────────────────────┐  │
│  │              │                │                │                      │  │
│  ▼              ▼                ▼                ▼                      ▼  │
│ ┌────┐      ┌─────┐         ┌─────────┐      ┌─────────┐         ┌───────┐ │
│ │SSM │      │ MCS │         │Predictor│      │Y-Encoder│         │ RAGAS │ │
│ └────┘      └─────┘         └─────────┘      └─────────┘         └───────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/prime/
├── __init__.py
├── prime.py                    # Main PRIME orchestration class
├── config.py                   # PRIMEConfig and component configs
├── exceptions.py               # API-specific exceptions
├── types.py                    # Shared type definitions
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application factory
│   ├── dependencies.py         # Dependency injection
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── process.py          # /process endpoint
│   │   ├── memory.py           # /memory/* endpoints
│   │   ├── diagnostics.py      # /diagnostics, /health
│   │   ├── clusters.py         # /clusters endpoints
│   │   └── config.py           # /config endpoint
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py             # API key authentication
│       ├── rate_limit.py       # Rate limiting
│       └── logging.py          # Request logging
├── adapters/
│   ├── __init__.py
│   ├── langchain.py            # LangChain retriever
│   └── llamaindex.py           # LlamaIndex retriever
├── evaluation/
│   ├── __init__.py
│   └── ragas.py                # RAGAS integration
└── client/
    ├── __init__.py
    └── client.py               # Python client SDK
```

### Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POST /process Flow                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  Client Request                                                    Response
       │                                                                ▲
       ▼                                                                │
┌──────────────┐                                                 ┌──────────────┐
│   Validate   │──▶ 400 Bad Request if invalid                  │   Format     │
│    Input     │                                                 │  Response    │
└──────┬───────┘                                                 └──────┬───────┘
       │                                                                │
       ▼                                                                │
┌──────────────┐                                                        │
│  Authenticate│──▶ 401 Unauthorized if invalid                        │
└──────┬───────┘                                                        │
       │                                                                │
       ▼                                                                │
┌──────────────┐                                                        │
│  Rate Limit  │──▶ 429 Too Many Requests if exceeded                  │
└──────┬───────┘                                                        │
       │                                                                │
       ▼                                                                │
┌──────────────────────────────────────────────────────────────────────┐│
│                         PRIME.process_turn()                          ││
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          ││
│  │  Y-Encoder   │────▶│     SSM      │────▶│  Decision    │          ││
│  │  encode(X)   │     │ update(S_X)  │     │  boundary?   │          ││
│  └──────┬───────┘     └──────────────┘     └──────┬───────┘          ││
│         │                                         │                   ││
│         │     ┌──────────────────────────────────┐│                   ││
│         │     │        Boundary Crossed          ││                   ││
│         │     └──────────────────────────────────┘│                   ││
│         │                    │                    │                   ││
│         ▼                    ▼                    ▼                   ││
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          ││
│  │ force_retrieval    │  Predictor   │     │  No Action   │──────────┼┤
│  │     OR SSM   │────▶│ predict(Ŝ_Y) │     │  (CONTINUE)  │          ││
│  │  triggered   │     └──────┬───────┘     └──────────────┘          ││
│  └──────────────┘            │                                        ││
│                              ▼                                        ││
│                       ┌──────────────┐                                ││
│                       │   MCS.search │                                ││
│                       │   (Ŝ_Y, k)   │────────────────────────────────┼┤
│                       └──────────────┘                                ││
└──────────────────────────────────────────────────────────────────────┘│
                                                                        │
                              ┌─────────────────────────────────────────┘
                              │
                              ▼
                        ProcessResponse
                        {
                          retrieved_memories: [...],
                          boundary_crossed: true,
                          variance: 0.23,
                          action: "retrieve"
                        }
```

---

## Technical Specification

### 1. Core Types and Exceptions

**File: `src/prime/types.py`**

```python
"""PRIME API type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActionState(StrEnum):
    """SSM action states."""

    CONTINUE = "continue"
    PREPARE = "prepare"
    RETRIEVE = "retrieve"
    RETRIEVE_CONSOLIDATE = "retrieve_consolidate"


@dataclass(frozen=True, slots=True)
class MemoryReadResult:
    """Memory retrieval result."""

    memory_id: str
    content: str
    cluster_id: int
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Memory write result."""

    memory_id: str
    cluster_id: int
    is_new_cluster: bool
    consolidated: bool


@dataclass(frozen=True, slots=True)
class PRIMEResponse:
    """Response from process_turn."""

    retrieved_memories: list[MemoryReadResult]
    boundary_crossed: bool
    variance: float
    smoothed_variance: float
    action: ActionState
    session_id: str
    turn_number: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class PRIMEDiagnostics:
    """System diagnostics."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    components: dict[str, ComponentStatus]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    """Individual component status."""

    name: str
    status: str
    latency_p50_ms: float
    error_rate: float
```

**File: `src/prime/exceptions.py`**

```python
"""PRIME API exceptions."""
from __future__ import annotations


class PRIMEError(Exception):
    """Base PRIME exception."""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ConfigurationError(PRIMEError):
    """Configuration validation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "PRIME_CONFIG_ERROR")


class ComponentError(PRIMEError):
    """Component initialization or operation failed."""

    def __init__(self, component: str, message: str) -> None:
        super().__init__(f"{component}: {message}", f"PRIME_{component.upper()}_ERROR")
        self.component = component


class SessionError(PRIMEError):
    """Session-related error."""

    def __init__(self, session_id: str, message: str) -> None:
        super().__init__(f"Session {session_id}: {message}", "PRIME_SESSION_ERROR")
        self.session_id = session_id


class AuthenticationError(PRIMEError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message, "PRIME_AUTH_ERROR")


class RateLimitError(PRIMEError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after}s",
            "PRIME_RATE_LIMIT_ERROR",
        )
        self.retry_after = retry_after
```

### 2. Configuration

**File: `src/prime/config.py`**

```python
"""PRIME configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from prime.encoder.config import YEncoderConfig
from prime.mcs.config import MCSConfig
from prime.predictor.config import PredictorConfig
from prime.ssm.config import SSMConfig


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=60, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = Field(default=30.0, ge=1.0)

    model_config = {"frozen": True}


class RAGASConfig(BaseModel):
    """RAGAS evaluation configuration."""

    enabled: bool = Field(default=True)
    llm_model: str = Field(default="gpt-4.1-mini")
    batch_size: int = Field(default=10, ge=1, le=100)
    timeout_seconds: float = Field(default=60.0, ge=1.0)

    model_config = {"frozen": True}


class PRIMEConfig(BaseModel):
    """Main PRIME system configuration."""

    # Component configs
    ssm: SSMConfig = Field(default_factory=SSMConfig)
    mcs: MCSConfig = Field(default_factory=MCSConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    y_encoder: YEncoderConfig = Field(default_factory=YEncoderConfig)

    # API config
    api: APIConfig = Field(default_factory=APIConfig)

    # Evaluation config
    ragas: RAGASConfig = Field(default_factory=RAGASConfig)

    # LLM integration
    llm_provider: str = Field(default="anthropic")
    llm_model: str = Field(default="claude-3.5-sonnet")

    model_config = {"frozen": True}

    @classmethod
    def from_env(cls) -> PRIMEConfig:
        """Load configuration from environment variables.

        Returns:
            PRIMEConfig with values from environment.

        Raises:
            ConfigurationError: If required variables missing.
        """
        import os

        from prime.exceptions import ConfigurationError

        # Build config from environment
        api_config = APIConfig(
            host=os.getenv("PRIME_HOST", "0.0.0.0"),
            port=int(os.getenv("PRIME_PORT", "8000")),
            workers=int(os.getenv("PRIME_WORKERS", "4")),
            rate_limit_per_minute=int(os.getenv("PRIME_RATE_LIMIT", "60")),
        )

        ragas_config = RAGASConfig(
            enabled=os.getenv("PRIME_RAGAS_ENABLED", "true").lower() == "true",
            llm_model=os.getenv("PRIME_RAGAS_MODEL", "gpt-4.1-mini"),
        )

        # Validate required environment variables
        required_vars = ["QDRANT_URL", "OPENAI_API_KEY"]
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            raise ConfigurationError(f"Missing environment variables: {missing}")

        return cls(api=api_config, ragas=ragas_config)
```

### 3. PRIME Orchestration Class

**File: `src/prime/prime.py`**

```python
"""PRIME orchestration class."""
from __future__ import annotations

import time
from typing import Any

import structlog

from prime.config import PRIMEConfig
from prime.encoder import YEncoder
from prime.exceptions import ComponentError, PRIMEError, SessionError
from prime.mcs import MemoryClusterStore
from prime.predictor import EmbeddingPredictor
from prime.ssm import SemanticStateMonitor
from prime.types import (
    ActionState,
    ComponentStatus,
    MemoryReadResult,
    MemoryWriteResult,
    PRIMEDiagnostics,
    PRIMEResponse,
)

logger = structlog.get_logger(__name__)


class PRIME:
    """Main PRIME orchestration class.

    Integrates SSM, Predictor, MCS, and Y-Encoder into a unified
    retrieval-augmented generation system.

    Example:
        ```python
        config = PRIMEConfig.from_env()
        prime = PRIME(config)

        response = prime.process_turn(
            input_text="What is PRIME?",
            session_id="sess_123",
        )
        ```
    """

    def __init__(self, config: PRIMEConfig) -> None:
        """Initialize PRIME with all components.

        Args:
            config: PRIME configuration.

        Raises:
            ComponentError: If any component fails to initialize.
        """
        self.config = config
        self._start_time = time.time()
        self._version = "1.0.0"

        # Initialize components
        self._init_components()

        logger.info(
            "prime_initialized",
            version=self._version,
            components=["ssm", "mcs", "predictor", "y_encoder"],
        )

    def _init_components(self) -> None:
        """Initialize all PRIME components."""
        try:
            self.y_encoder = YEncoder(self.config.y_encoder)
        except Exception as e:
            raise ComponentError("y_encoder", str(e)) from e

        try:
            self.ssm = SemanticStateMonitor(self.config.ssm, encoder=self.y_encoder)
        except Exception as e:
            raise ComponentError("ssm", str(e)) from e

        try:
            self.mcs = MemoryClusterStore(self.config.mcs, encoder=self.y_encoder)
        except Exception as e:
            raise ComponentError("mcs", str(e)) from e

        try:
            self.predictor = EmbeddingPredictor(self.config.predictor)
        except Exception as e:
            raise ComponentError("predictor", str(e)) from e

    def process_turn(
        self,
        input_text: str,
        session_id: str,
        force_retrieval: bool = False,
        k: int = 5,
    ) -> PRIMEResponse:
        """Process a conversation turn through the PRIME pipeline.

        This is the main entry point for RAG processing. It:
        1. Encodes input to embedding
        2. Updates SSM state and checks for boundary crossing
        3. If boundary crossed or forced: predicts target and retrieves
        4. Returns retrieved memories and diagnostics

        Args:
            input_text: User input text.
            session_id: Session identifier for SSM state.
            force_retrieval: If True, bypass SSM and force retrieval.
            k: Number of memories to retrieve.

        Returns:
            PRIMEResponse with retrieved memories and diagnostics.

        Raises:
            SessionError: If session state is invalid.
            ComponentError: If any component fails.
        """
        start = time.perf_counter()

        logger.info(
            "process_turn_start",
            session_id=session_id,
            input_length=len(input_text),
            force_retrieval=force_retrieval,
        )

        # Step 1: Encode input
        input_embedding = self.y_encoder.encode(input_text)

        # Step 2: Update SSM
        ssm_result = self.ssm.update(
            session_id=session_id,
            embedding=input_embedding,
        )

        # Step 3: Decide retrieval
        should_retrieve = force_retrieval or ssm_result.action in (
            ActionState.RETRIEVE,
            ActionState.RETRIEVE_CONSOLIDATE,
        )

        retrieved_memories: list[MemoryReadResult] = []

        if should_retrieve:
            # Step 4a: Predict target embedding
            context_embeddings = self.ssm.get_context_window(session_id)
            predicted_embedding = self.predictor.predict(
                context=context_embeddings,
                current=input_embedding,
            )

            # Step 4b: Search MCS
            search_results = self.mcs.search(
                query_embedding=predicted_embedding,
                k=k,
                session_id=session_id,
            )
            retrieved_memories = [
                MemoryReadResult(
                    memory_id=r.memory_id,
                    content=r.content,
                    cluster_id=r.cluster_id,
                    similarity=r.similarity,
                    metadata=r.metadata,
                    created_at=r.created_at,
                )
                for r in search_results
            ]

            # Step 4c: Consolidate if needed
            if ssm_result.action == ActionState.RETRIEVE_CONSOLIDATE:
                self.mcs.consolidate(session_id=session_id)

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "process_turn_complete",
            session_id=session_id,
            action=ssm_result.action,
            retrieved_count=len(retrieved_memories),
            latency_ms=latency_ms,
        )

        return PRIMEResponse(
            retrieved_memories=retrieved_memories,
            boundary_crossed=ssm_result.action != ActionState.CONTINUE,
            variance=ssm_result.variance,
            smoothed_variance=ssm_result.smoothed_variance,
            action=ssm_result.action,
            session_id=session_id,
            turn_number=ssm_result.turn_number,
            latency_ms=latency_ms,
        )

    def record_response(
        self,
        response: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Record LLM response to memory.

        Call this after generating an LLM response to store it
        in PRIME memory for future retrieval.

        Args:
            response: LLM response text.
            session_id: Session identifier.
            metadata: Optional metadata.

        Returns:
            MemoryWriteResult with cluster assignment.
        """
        logger.info(
            "record_response",
            session_id=session_id,
            response_length=len(response),
        )

        # Encode and store
        embedding = self.y_encoder.encode(response)
        result = self.mcs.write(
            content=response,
            embedding=embedding,
            session_id=session_id,
            metadata=metadata or {},
        )

        return MemoryWriteResult(
            memory_id=result.memory_id,
            cluster_id=result.cluster_id,
            is_new_cluster=result.is_new_cluster,
            consolidated=result.consolidated,
        )

    def write_external_knowledge(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Write external knowledge to memory.

        Use this for document ingestion or adding knowledge
        that doesn't come from conversation.

        Args:
            content: Knowledge content.
            metadata: Optional metadata.

        Returns:
            MemoryWriteResult with cluster assignment.
        """
        logger.info("write_external_knowledge", content_length=len(content))

        embedding = self.y_encoder.encode(content)
        result = self.mcs.write(
            content=content,
            embedding=embedding,
            metadata=metadata or {"source": "external"},
        )

        return MemoryWriteResult(
            memory_id=result.memory_id,
            cluster_id=result.cluster_id,
            is_new_cluster=result.is_new_cluster,
            consolidated=result.consolidated,
        )

    def search_memory(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[MemoryReadResult]:
        """Direct memory search without SSM.

        Args:
            query: Query text.
            k: Number of results.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of matching memories.
        """
        embedding = self.y_encoder.encode(query)
        results = self.mcs.search(
            query_embedding=embedding,
            k=k,
            min_similarity=min_similarity,
        )

        return [
            MemoryReadResult(
                memory_id=r.memory_id,
                content=r.content,
                cluster_id=r.cluster_id,
                similarity=r.similarity,
                metadata=r.metadata,
                created_at=r.created_at,
            )
            for r in results
        ]

    def get_diagnostics(self) -> PRIMEDiagnostics:
        """Get system diagnostics.

        Returns:
            PRIMEDiagnostics with component health and metrics.
        """
        uptime = time.time() - self._start_time

        # Collect component status
        components = {
            "ssm": ComponentStatus(
                name="SemanticStateMonitor",
                status="healthy",
                latency_p50_ms=self.ssm.get_latency_p50(),
                error_rate=self.ssm.get_error_rate(),
            ),
            "mcs": ComponentStatus(
                name="MemoryClusterStore",
                status="healthy",
                latency_p50_ms=self.mcs.get_latency_p50(),
                error_rate=self.mcs.get_error_rate(),
            ),
            "predictor": ComponentStatus(
                name="EmbeddingPredictor",
                status="healthy",
                latency_p50_ms=self.predictor.get_latency_p50(),
                error_rate=self.predictor.get_error_rate(),
            ),
            "y_encoder": ComponentStatus(
                name="YEncoder",
                status="healthy",
                latency_p50_ms=self.y_encoder.get_latency_p50(),
                error_rate=self.y_encoder.get_error_rate(),
            ),
        }

        # Determine overall status
        error_rates = [c.error_rate for c in components.values()]
        if max(error_rates) > 0.1:
            status = "unhealthy"
        elif max(error_rates) > 0.01:
            status = "degraded"
        else:
            status = "healthy"

        return PRIMEDiagnostics(
            status=status,
            version=self._version,
            uptime_seconds=uptime,
            components=components,
            metrics={
                "total_sessions": self.ssm.get_session_count(),
                "total_memories": self.mcs.get_memory_count(),
                "cluster_count": self.mcs.get_cluster_count(),
            },
        )
```

### 4. FastAPI Application

**File: `src/prime/api/app.py`**

```python
"""FastAPI application factory."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prime.api.dependencies import get_prime, init_prime
from prime.api.middleware.auth import APIKeyMiddleware
from prime.api.middleware.rate_limit import RateLimitMiddleware
from prime.api.routes import clusters, config, diagnostics, memory, process
from prime.config import PRIMEConfig
from prime.exceptions import PRIMEError

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    logger.info("prime_api_starting")

    # Initialize PRIME
    config = PRIMEConfig.from_env()
    init_prime(config)

    logger.info("prime_api_ready")
    yield

    logger.info("prime_api_shutdown")


def create_app(config: PRIMEConfig | None = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        config: Optional PRIME configuration.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="PRIME API",
        description="Predictive Retrieval with Integrated Memory Engine",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Add routes
    app.include_router(process.router, tags=["Process"])
    app.include_router(memory.router, prefix="/memory", tags=["Memory"])
    app.include_router(clusters.router, prefix="/clusters", tags=["Clusters"])
    app.include_router(diagnostics.router, tags=["Diagnostics"])
    app.include_router(config.router, prefix="/config", tags=["Config"])

    # Exception handlers
    @app.exception_handler(PRIMEError)
    async def prime_error_handler(request: Request, exc: PRIMEError) -> JSONResponse:
        """Handle PRIME exceptions."""
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.message,
                "error_code": exc.error_code,
                "request_id": request.state.request_id,
            },
        )

    # Request timing middleware
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        """Add request timing and ID."""
        import uuid

        request.state.request_id = str(uuid.uuid4())
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time-Ms"] = str(int(duration * 1000))

        return response

    return app


# For uvicorn
app = create_app()
```

### 5. Route Implementations

**File: `src/prime/api/routes/process.py`**

```python
"""Process route - main conversation endpoint."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from prime.api.dependencies import get_prime
from prime.prime import PRIME
from prime.types import ActionState, MemoryReadResult

router = APIRouter()


class ProcessRequest(BaseModel):
    """Request for processing a conversation turn."""

    input: str = Field(min_length=1, max_length=8192)
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str | None = None
    force_retrieval: bool = False
    k: int = Field(default=5, ge=1, le=20)

    model_config = {"frozen": True}


class MemoryResult(BaseModel):
    """Memory result in API response."""

    memory_id: str
    content: str
    cluster_id: int
    similarity: float
    metadata: dict

    @classmethod
    def from_internal(cls, result: MemoryReadResult) -> MemoryResult:
        """Convert from internal type."""
        return cls(
            memory_id=result.memory_id,
            content=result.content,
            cluster_id=result.cluster_id,
            similarity=result.similarity,
            metadata=result.metadata,
        )


class ProcessResponse(BaseModel):
    """Response from process_turn."""

    retrieved_memories: list[MemoryResult]
    boundary_crossed: bool
    variance: float
    smoothed_variance: float
    action: ActionState
    session_id: str
    turn_number: int
    latency_ms: float


@router.post("/process", response_model=ProcessResponse)
async def process_turn(
    request: ProcessRequest,
    prime: Annotated[PRIME, Depends(get_prime)],
) -> ProcessResponse:
    """Process a conversation turn through PRIME.

    This endpoint encodes the input, updates the semantic state,
    and retrieves relevant memories if a boundary is crossed.

    Args:
        request: Process request with input and session.
        prime: PRIME instance from dependency injection.

    Returns:
        ProcessResponse with retrieved memories and diagnostics.
    """
    result = prime.process_turn(
        input_text=request.input,
        session_id=request.session_id,
        force_retrieval=request.force_retrieval,
        k=request.k,
    )

    return ProcessResponse(
        retrieved_memories=[
            MemoryResult.from_internal(m) for m in result.retrieved_memories
        ],
        boundary_crossed=result.boundary_crossed,
        variance=result.variance,
        smoothed_variance=result.smoothed_variance,
        action=result.action,
        session_id=result.session_id,
        turn_number=result.turn_number,
        latency_ms=result.latency_ms,
    )
```

**File: `src/prime/api/routes/memory.py`**

```python
"""Memory routes - write and search endpoints."""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from prime.api.dependencies import get_prime
from prime.prime import PRIME

router = APIRouter()


class MemoryWriteRequest(BaseModel):
    """Request to write memory."""

    content: str = Field(min_length=1, max_length=50000)
    metadata: dict[str, str | int | float | bool] | None = None
    user_id: str | None = None
    session_id: str | None = None

    model_config = {"frozen": True}


class MemoryWriteResponse(BaseModel):
    """Response from memory write."""

    memory_id: str
    cluster_id: int
    is_new_cluster: bool
    consolidated: bool


class MemorySearchRequest(BaseModel):
    """Request for direct memory search."""

    query: str = Field(min_length=1, max_length=8192)
    k: int = Field(default=5, ge=1, le=100)
    user_id: str | None = None
    session_id: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class MemorySearchResponse(BaseModel):
    """Response from memory search."""

    results: list[dict[str, Any]]
    count: int


@router.post("/write", response_model=MemoryWriteResponse)
async def write_memory(
    request: MemoryWriteRequest,
    prime: Annotated[PRIME, Depends(get_prime)],
) -> MemoryWriteResponse:
    """Write external knowledge to PRIME memory.

    Use this endpoint to ingest documents or add knowledge
    that doesn't come from conversation.
    """
    result = prime.write_external_knowledge(
        content=request.content,
        metadata=request.metadata,
    )

    return MemoryWriteResponse(
        memory_id=result.memory_id,
        cluster_id=result.cluster_id,
        is_new_cluster=result.is_new_cluster,
        consolidated=result.consolidated,
    )


@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    prime: Annotated[PRIME, Depends(get_prime)],
) -> MemorySearchResponse:
    """Search PRIME memory directly.

    This bypasses SSM and searches memory directly using
    the provided query.
    """
    results = prime.search_memory(
        query=request.query,
        k=request.k,
        min_similarity=request.min_similarity,
    )

    return MemorySearchResponse(
        results=[
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "cluster_id": r.cluster_id,
                "similarity": r.similarity,
                "metadata": r.metadata,
            }
            for r in results
        ],
        count=len(results),
    )
```

### 6. Authentication Middleware

**File: `src/prime/api/middleware/auth.py`**

```python
"""API key authentication middleware."""
from __future__ import annotations

import os
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Endpoints that don't require authentication
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Validate API key for protected endpoints."""
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Get expected API key from environment
        expected_key = os.getenv("PRIME_API_KEY")
        if not expected_key:
            # No key configured = no auth required (dev mode)
            return await call_next(request)

        # Get provided key from header
        header_name = os.getenv("PRIME_API_KEY_HEADER", "X-API-Key")
        provided_key = request.headers.get(header_name)

        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "API key required",
                    "error_code": "PRIME_AUTH_ERROR",
                    "detail": f"Provide API key in {header_name} header",
                },
            )

        if provided_key != expected_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid API key",
                    "error_code": "PRIME_AUTH_ERROR",
                },
            )

        return await call_next(request)
```

### 7. Rate Limiting Middleware

**File: `src/prime/api/middleware/rate_limit.py`**

```python
"""Rate limiting middleware."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiting middleware."""

    def __init__(self, app, window_seconds: int = 60) -> None:
        """Initialize rate limiter.

        Args:
            app: ASGI application.
            window_seconds: Rate limit window in seconds.
        """
        super().__init__(app)
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use API key if present, otherwise IP
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{api_key[:8]}"  # Use prefix only
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def _cleanup_old_requests(self, client_id: str, now: float) -> None:
        """Remove requests outside the window."""
        cutoff = now - self.window_seconds
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > cutoff
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Apply rate limiting."""
        # Get rate limit from environment
        limit = int(os.getenv("PRIME_RATE_LIMIT", "60"))
        if limit <= 0:
            return await call_next(request)

        client_id = self._get_client_id(request)
        now = time.time()

        # Cleanup old requests
        self._cleanup_old_requests(client_id, now)

        # Check limit
        if len(self.requests[client_id]) >= limit:
            retry_after = int(
                self.window_seconds
                - (now - min(self.requests[client_id]))
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "PRIME_RATE_LIMIT_ERROR",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Record request
        self.requests[client_id].append(now)

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(
            limit - len(self.requests[client_id])
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(now + self.window_seconds)
        )

        return response
```

### 8. RAGAS Evaluation

**File: `src/prime/evaluation/ragas.py`**

```python
"""RAGAS evaluation integration."""
from __future__ import annotations

import time
from typing import Any

import structlog
from datasets import Dataset
from pydantic import BaseModel, Field
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from prime.config import RAGASConfig
from prime.exceptions import ComponentError

logger = structlog.get_logger(__name__)


class EvaluationRequest(BaseModel):
    """Request for RAGAS evaluation."""

    query: str = Field(min_length=1, max_length=8192)
    retrieved_contexts: list[str] = Field(min_length=1, max_length=20)
    generated_response: str = Field(min_length=1, max_length=50000)
    ground_truth: str | None = Field(
        default=None,
        description="Optional ground truth for reference-based metrics",
    )

    model_config = {"frozen": True}


class EvaluationResponse(BaseModel):
    """Response with RAGAS evaluation metrics."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float


class RAGASEvaluator:
    """RAGAS evaluation service for PRIME.

    Provides reference-free metrics for RAG quality assessment:
    - Faithfulness: Factual consistency with retrieved contexts
    - Context Precision: Relevance of retrieved contexts
    - Answer Relevancy: Relevance of response to query
    - Context Recall: Coverage of ground truth (requires reference)
    """

    def __init__(self, config: RAGASConfig) -> None:
        """Initialize RAGAS evaluator.

        Args:
            config: RAGAS configuration.

        Raises:
            ComponentError: If LLM setup fails.
        """
        self.config = config
        self.metrics = [
            faithfulness,
            context_precision,
            answer_relevancy,
        ]

        if config.enabled:
            self._setup_llm()

        logger.info(
            "ragas_evaluator_initialized",
            enabled=config.enabled,
            model=config.llm_model,
        )

    def _setup_llm(self) -> None:
        """Configure LLM for RAGAS evaluation."""
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=0,
            )

            # Set LLM for all metrics
            for metric in self.metrics:
                metric.llm = self.llm

        except Exception as e:
            raise ComponentError("ragas", f"LLM setup failed: {e}") from e

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

        Raises:
            ComponentError: If evaluation fails.
        """
        if not self.config.enabled:
            raise ComponentError("ragas", "Evaluation is disabled")

        start = time.perf_counter()

        # Prepare dataset
        data: dict[str, Any] = {
            "question": [query],
            "contexts": [retrieved_contexts],
            "answer": [generated_response],
        }

        metrics = self.metrics.copy()
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        try:
            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=metrics)
        except Exception as e:
            raise ComponentError("ragas", f"Evaluation failed: {e}") from e

        latency_ms = (time.perf_counter() - start) * 1000

        # Extract scores
        scores: dict[str, float] = {
            "faithfulness": float(result["faithfulness"]),
            "context_precision": float(result["context_precision"]),
            "answer_relevancy": float(result["answer_relevancy"]),
        }

        context_recall_score = None
        if ground_truth:
            context_recall_score = float(result["context_recall"])
            scores["context_recall"] = context_recall_score

        overall = sum(scores.values()) / len(scores)

        logger.info(
            "ragas_evaluation_complete",
            overall_score=overall,
            latency_ms=latency_ms,
        )

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
        samples: list[dict[str, Any]],
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

### 9. LangChain Adapter

**File: `src/prime/adapters/langchain.py`**

```python
"""LangChain retriever adapter for PRIME."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

if TYPE_CHECKING:
    from prime.client import PRIMEClient


class PRIMERetriever(BaseRetriever):
    """PRIME as a LangChain retriever.

    Enables using PRIME within LangChain chains and agents.

    Example:
        ```python
        from prime.adapters.langchain import PRIMERetriever
        from prime.client import PRIMEClient

        client = PRIMEClient(
            base_url="http://localhost:8000",
            api_key="your-api-key",
        )

        retriever = PRIMERetriever(
            prime_client=client,
            k=5,
        )

        # Use in a chain
        from langchain_core.runnables import RunnablePassthrough

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        result = chain.invoke("What is PRIME?")
        ```
    """

    prime_client: Any = Field(description="PRIME client instance")
    k: int = Field(default=5, description="Number of documents to retrieve")
    session_id: str | None = Field(
        default=None,
        description="Session ID for stateful retrieval",
    )
    use_process_turn: bool = Field(
        default=False,
        description="Use process_turn (with SSM) vs direct search",
    )

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
        if self.use_process_turn and self.session_id:
            # Use full PRIME pipeline with SSM
            result = self.prime_client.process_turn(
                input_text=query,
                session_id=self.session_id,
                k=self.k,
            )
            memories = result.retrieved_memories
        else:
            # Direct search without SSM
            result = self.prime_client.search(
                query=query,
                k=self.k,
            )
            memories = result.results

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
            for memory in memories
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Async retrieve relevant documents from PRIME."""
        # For now, use sync implementation
        # TODO: Implement async client
        return self._get_relevant_documents(query, run_manager=run_manager)
```

### 10. Python Client SDK

**File: `src/prime/client/client.py`**

```python
"""PRIME Python client SDK."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class MemoryResult:
    """Memory retrieval result."""

    memory_id: str
    content: str
    cluster_id: int
    similarity: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Result from process_turn."""

    retrieved_memories: list[MemoryResult]
    boundary_crossed: bool
    variance: float
    action: str
    session_id: str
    turn_number: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result from direct search."""

    results: list[MemoryResult]
    count: int


class PRIMEClient:
    """Python client for PRIME API.

    Example:
        ```python
        client = PRIMEClient(
            base_url="http://localhost:8000",
            api_key="your-api-key",
        )

        # Process conversation turn
        result = client.process_turn(
            input_text="What is PRIME?",
            session_id="sess_123",
        )

        for memory in result.retrieved_memories:
            print(f"{memory.content} (similarity: {memory.similarity})")

        # Direct search
        search = client.search("PRIME architecture")
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize PRIME client.

        Args:
            base_url: PRIME API base URL.
            api_key: Optional API key.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    def process_turn(
        self,
        input_text: str,
        session_id: str,
        force_retrieval: bool = False,
        k: int = 5,
    ) -> ProcessResult:
        """Process a conversation turn through PRIME.

        Args:
            input_text: User input text.
            session_id: Session identifier.
            force_retrieval: Force retrieval regardless of SSM.
            k: Number of memories to retrieve.

        Returns:
            ProcessResult with retrieved memories.

        Raises:
            httpx.HTTPStatusError: If request fails.
        """
        response = self._client.post(
            "/process",
            json={
                "input": input_text,
                "session_id": session_id,
                "force_retrieval": force_retrieval,
                "k": k,
            },
        )
        response.raise_for_status()
        data = response.json()

        return ProcessResult(
            retrieved_memories=[
                MemoryResult(
                    memory_id=m["memory_id"],
                    content=m["content"],
                    cluster_id=m["cluster_id"],
                    similarity=m["similarity"],
                    metadata=m["metadata"],
                )
                for m in data["retrieved_memories"]
            ],
            boundary_crossed=data["boundary_crossed"],
            variance=data["variance"],
            action=data["action"],
            session_id=data["session_id"],
            turn_number=data["turn_number"],
            latency_ms=data["latency_ms"],
        )

    def search(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> SearchResult:
        """Search PRIME memory directly.

        Args:
            query: Query text.
            k: Number of results.
            min_similarity: Minimum similarity threshold.

        Returns:
            SearchResult with matching memories.
        """
        response = self._client.post(
            "/memory/search",
            json={
                "query": query,
                "k": k,
                "min_similarity": min_similarity,
            },
        )
        response.raise_for_status()
        data = response.json()

        return SearchResult(
            results=[
                MemoryResult(
                    memory_id=r["memory_id"],
                    content=r["content"],
                    cluster_id=r["cluster_id"],
                    similarity=r["similarity"],
                    metadata=r["metadata"],
                )
                for r in data["results"]
            ],
            count=data["count"],
        )

    def write_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write external knowledge to PRIME memory.

        Args:
            content: Content to write.
            metadata: Optional metadata.

        Returns:
            Write result with memory and cluster IDs.
        """
        response = self._client.post(
            "/memory/write",
            json={
                "content": content,
                "metadata": metadata or {},
            },
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, str]:
        """Check API health.

        Returns:
            Health status dict.
        """
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self) -> PRIMEClient:
        """Context manager enter."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
```

---

## Test Specification

### Test Structure

```
tests/
├── unit/
│   ├── test_prime.py           # PRIME class tests
│   ├── test_config.py          # Configuration tests
│   ├── test_types.py           # Type tests
│   └── api/
│       ├── test_routes.py      # Route unit tests
│       └── test_middleware.py  # Middleware tests
├── integration/
│   ├── test_api.py             # Full API integration
│   ├── test_client.py          # Client SDK tests
│   └── test_adapters.py        # Adapter tests
└── fixtures/
    └── conftest.py             # Shared fixtures
```

### Unit Tests

**File: `tests/unit/test_prime.py`**

```python
"""Unit tests for PRIME orchestration class."""
from __future__ import annotations

import pytest

from prime.config import PRIMEConfig
from prime.exceptions import ComponentError, SessionError
from prime.prime import PRIME
from prime.types import ActionState


class TestPRIMEInit:
    """Tests for PRIME initialization."""

    def test_init_success(self, mock_components: dict) -> None:
        """Test successful PRIME initialization."""
        config = PRIMEConfig()
        prime = PRIME(config)

        assert prime.config == config
        assert prime.y_encoder is not None
        assert prime.ssm is not None
        assert prime.mcs is not None
        assert prime.predictor is not None

    def test_init_component_error(self, mock_failing_encoder: None) -> None:
        """Test initialization with failing component."""
        config = PRIMEConfig()

        with pytest.raises(ComponentError) as exc_info:
            PRIME(config)

        assert "y_encoder" in exc_info.value.component


class TestProcessTurn:
    """Tests for process_turn method."""

    def test_process_turn_no_retrieval(
        self,
        prime: PRIME,
        mock_ssm_continue: None,
    ) -> None:
        """Test process_turn when no retrieval needed."""
        result = prime.process_turn(
            input_text="Hello",
            session_id="test_session",
        )

        assert result.boundary_crossed is False
        assert result.action == ActionState.CONTINUE
        assert len(result.retrieved_memories) == 0
        assert result.session_id == "test_session"
        assert result.latency_ms > 0

    def test_process_turn_with_retrieval(
        self,
        prime: PRIME,
        mock_ssm_retrieve: None,
        mock_predictor: None,
        mock_mcs: None,
    ) -> None:
        """Test process_turn when retrieval triggered."""
        result = prime.process_turn(
            input_text="What is PRIME?",
            session_id="test_session",
        )

        assert result.boundary_crossed is True
        assert result.action == ActionState.RETRIEVE
        assert len(result.retrieved_memories) > 0
        assert result.retrieved_memories[0].content is not None

    def test_process_turn_force_retrieval(
        self,
        prime: PRIME,
        mock_ssm_continue: None,
        mock_predictor: None,
        mock_mcs: None,
    ) -> None:
        """Test force_retrieval bypasses SSM."""
        result = prime.process_turn(
            input_text="Hello",
            session_id="test_session",
            force_retrieval=True,
        )

        # Even though SSM says continue, we force retrieval
        assert len(result.retrieved_memories) > 0

    def test_process_turn_consolidation(
        self,
        prime: PRIME,
        mock_ssm_consolidate: None,
        mock_predictor: None,
        mock_mcs: None,
    ) -> None:
        """Test retrieval with consolidation."""
        result = prime.process_turn(
            input_text="Long conversation continues...",
            session_id="test_session",
        )

        assert result.action == ActionState.RETRIEVE_CONSOLIDATE


class TestRecordResponse:
    """Tests for record_response method."""

    def test_record_response(
        self,
        prime: PRIME,
        mock_mcs: None,
    ) -> None:
        """Test recording LLM response."""
        result = prime.record_response(
            response="The answer is 42.",
            session_id="test_session",
        )

        assert result.memory_id is not None
        assert result.cluster_id >= 0

    def test_record_response_with_metadata(
        self,
        prime: PRIME,
        mock_mcs: None,
    ) -> None:
        """Test recording with metadata."""
        result = prime.record_response(
            response="Test response",
            session_id="test_session",
            metadata={"model": "claude-3.5-sonnet"},
        )

        assert result.memory_id is not None


class TestWriteExternalKnowledge:
    """Tests for write_external_knowledge method."""

    def test_write_external_knowledge(
        self,
        prime: PRIME,
        mock_mcs: None,
    ) -> None:
        """Test writing external knowledge."""
        result = prime.write_external_knowledge(
            content="PRIME is an intelligent memory system.",
            metadata={"source": "documentation"},
        )

        assert result.memory_id is not None
        assert result.cluster_id >= 0


class TestSearchMemory:
    """Tests for search_memory method."""

    def test_search_memory(
        self,
        prime: PRIME,
        mock_mcs: None,
    ) -> None:
        """Test direct memory search."""
        results = prime.search_memory(
            query="PRIME architecture",
            k=5,
        )

        assert len(results) <= 5
        for result in results:
            assert result.content is not None
            assert result.similarity >= 0.0


class TestGetDiagnostics:
    """Tests for get_diagnostics method."""

    def test_get_diagnostics(self, prime: PRIME) -> None:
        """Test diagnostics retrieval."""
        diagnostics = prime.get_diagnostics()

        assert diagnostics.status in ("healthy", "degraded", "unhealthy")
        assert diagnostics.version is not None
        assert diagnostics.uptime_seconds >= 0
        assert "ssm" in diagnostics.components
        assert "mcs" in diagnostics.components
```

### Integration Tests

**File: `tests/integration/test_api.py`**

```python
"""Integration tests for PRIME API."""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from prime.api.app import create_app


@pytest.fixture
async def client() -> AsyncClient:
    """Create async test client."""
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestProcessEndpoint:
    """Tests for POST /process."""

    @pytest.mark.asyncio
    async def test_process_success(self, client: AsyncClient) -> None:
        """Test successful process request."""
        response = await client.post(
            "/process",
            json={
                "input": "What is PRIME?",
                "session_id": "test_session",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "retrieved_memories" in data
        assert "boundary_crossed" in data
        assert "action" in data

    @pytest.mark.asyncio
    async def test_process_validation_error(
        self,
        client: AsyncClient,
    ) -> None:
        """Test validation error for empty input."""
        response = await client.post(
            "/process",
            json={
                "input": "",
                "session_id": "test_session",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_process_with_force_retrieval(
        self,
        client: AsyncClient,
    ) -> None:
        """Test process with force_retrieval=True."""
        response = await client.post(
            "/process",
            json={
                "input": "Hello",
                "session_id": "test_session",
                "force_retrieval": True,
            },
        )

        assert response.status_code == 200
        # Should have memories even for simple input


class TestMemoryEndpoints:
    """Tests for /memory/* endpoints."""

    @pytest.mark.asyncio
    async def test_write_memory(self, client: AsyncClient) -> None:
        """Test memory write."""
        response = await client.post(
            "/memory/write",
            json={
                "content": "Test knowledge content",
                "metadata": {"source": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "memory_id" in data
        assert "cluster_id" in data

    @pytest.mark.asyncio
    async def test_search_memory(self, client: AsyncClient) -> None:
        """Test memory search."""
        response = await client.post(
            "/memory/search",
            json={
                "query": "test query",
                "k": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data


class TestDiagnosticsEndpoints:
    """Tests for diagnostics endpoints."""

    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient) -> None:
        """Test health endpoint."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_diagnostics(self, client: AsyncClient) -> None:
        """Test diagnostics endpoint."""
        response = await client.get("/diagnostics")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "metrics" in data


class TestAuthentication:
    """Tests for API authentication."""

    @pytest.mark.asyncio
    async def test_missing_api_key(
        self,
        client_with_auth: AsyncClient,
    ) -> None:
        """Test request without API key."""
        response = await client_with_auth.post(
            "/process",
            json={
                "input": "Test",
                "session_id": "test",
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_api_key(
        self,
        client_with_auth: AsyncClient,
    ) -> None:
        """Test request with invalid API key."""
        response = await client_with_auth.post(
            "/process",
            json={
                "input": "Test",
                "session_id": "test",
            },
            headers={"X-API-Key": "invalid"},
        )

        assert response.status_code == 401


class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_headers(
        self,
        client: AsyncClient,
    ) -> None:
        """Test rate limit headers in response."""
        response = await client.get("/health")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(
        self,
        client_low_limit: AsyncClient,
    ) -> None:
        """Test rate limit exceeded response."""
        # Make requests until limit exceeded
        for _ in range(10):
            response = await client_low_limit.get("/health")
            if response.status_code == 429:
                break

        assert response.status_code == 429
        assert "Retry-After" in response.headers
```

### RAGAS Tests

**File: `tests/integration/test_ragas.py`**

```python
"""Tests for RAGAS evaluation integration."""
from __future__ import annotations

import pytest

from prime.config import RAGASConfig
from prime.evaluation.ragas import EvaluationRequest, RAGASEvaluator


@pytest.fixture
def evaluator() -> RAGASEvaluator:
    """Create RAGAS evaluator."""
    config = RAGASConfig(enabled=True, llm_model="gpt-4.1-mini")
    return RAGASEvaluator(config)


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator."""

    @pytest.mark.slow
    def test_evaluate_faithfulness(self, evaluator: RAGASEvaluator) -> None:
        """Test faithfulness evaluation."""
        result = evaluator.evaluate(
            query="What is the capital of France?",
            retrieved_contexts=["Paris is the capital of France."],
            generated_response="The capital of France is Paris.",
        )

        assert 0.0 <= result.faithfulness <= 1.0
        assert result.faithfulness > 0.8  # Should be high for faithful response

    @pytest.mark.slow
    def test_evaluate_context_precision(
        self,
        evaluator: RAGASEvaluator,
    ) -> None:
        """Test context precision evaluation."""
        result = evaluator.evaluate(
            query="What is Python?",
            retrieved_contexts=[
                "Python is a programming language.",
                "Snakes are reptiles.",  # Irrelevant
            ],
            generated_response="Python is a programming language.",
        )

        assert 0.0 <= result.context_precision <= 1.0

    @pytest.mark.slow
    def test_evaluate_answer_relevancy(
        self,
        evaluator: RAGASEvaluator,
    ) -> None:
        """Test answer relevancy evaluation."""
        result = evaluator.evaluate(
            query="What is 2+2?",
            retrieved_contexts=["Basic arithmetic operations."],
            generated_response="2+2 equals 4.",
        )

        assert 0.0 <= result.answer_relevancy <= 1.0
        assert result.answer_relevancy > 0.8

    @pytest.mark.slow
    def test_evaluate_with_ground_truth(
        self,
        evaluator: RAGASEvaluator,
    ) -> None:
        """Test evaluation with ground truth for context recall."""
        result = evaluator.evaluate(
            query="What is PRIME?",
            retrieved_contexts=[
                "PRIME is an intelligent memory system for RAG.",
            ],
            generated_response="PRIME is a memory system.",
            ground_truth="PRIME is a Predictive Retrieval with Integrated Memory Engine.",
        )

        assert result.context_recall is not None
        assert 0.0 <= result.context_recall <= 1.0

    @pytest.mark.slow
    def test_evaluate_overall_score(
        self,
        evaluator: RAGASEvaluator,
    ) -> None:
        """Test overall score calculation."""
        result = evaluator.evaluate(
            query="What is Python?",
            retrieved_contexts=["Python is a programming language."],
            generated_response="Python is a programming language.",
        )

        # Overall should be average of metrics
        expected = (
            result.faithfulness
            + result.context_precision
            + result.answer_relevancy
        ) / 3

        assert abs(result.overall_score - expected) < 0.01
```

---

## Implementation Roadmap

### Phase 1: Core PRIME Class (Week 1)

| Task | Priority | Effort |
|------|----------|--------|
| Create project structure | P0 | S |
| Implement types.py | P0 | S |
| Implement exceptions.py | P0 | S |
| Implement config.py | P0 | M |
| Implement prime.py | P0 | L |
| Unit tests for PRIME class | P0 | M |

### Phase 2: FastAPI Application (Week 2)

| Task | Priority | Effort |
|------|----------|--------|
| Create app.py factory | P0 | M |
| Implement process route | P0 | M |
| Implement memory routes | P0 | M |
| Implement diagnostics routes | P1 | S |
| Implement clusters routes | P1 | S |
| Integration tests | P0 | M |

### Phase 3: Middleware & Security (Week 3)

| Task | Priority | Effort |
|------|----------|--------|
| Implement auth middleware | P0 | M |
| Implement rate limiting | P0 | M |
| Add request logging | P1 | S |
| Add CORS configuration | P1 | S |
| Security testing | P0 | M |

### Phase 4: Adapters & Evaluation (Week 4)

| Task | Priority | Effort |
|------|----------|--------|
| Implement LangChain adapter | P2 | M |
| Implement LlamaIndex adapter | P2 | M |
| Implement RAGAS evaluator | P1 | L |
| Add evaluation endpoint | P1 | M |
| Adapter tests | P2 | M |

### Phase 5: Client SDK & Documentation (Week 5)

| Task | Priority | Effort |
|------|----------|--------|
| Implement Python client | P1 | M |
| Client SDK tests | P1 | M |
| OpenAPI documentation | P1 | S |
| Usage examples | P1 | M |
| Performance testing | P1 | M |

---

## Risk Management

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Component integration failures | High | Mock-based unit tests, integration tests |
| Latency exceeds targets | High | Performance profiling, caching |
| Rate limiter memory growth | Medium | Time-based cleanup, Redis backend |
| RAGAS LLM costs | Medium | Batch evaluation, sampling |

### Dependencies

| Dependency | Risk Level | Fallback |
|------------|------------|----------|
| FastAPI | Low | Well-maintained, stable |
| RAGAS | Medium | Can disable evaluation |
| Redis (rate limiting) | Low | In-memory fallback |
| LangChain | Medium | Optional adapter |

---

## Appendix: API Reference

### OpenAPI Specification

The API provides auto-generated OpenAPI documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| PRIME_AUTH_ERROR | 401 | Authentication failed |
| PRIME_RATE_LIMIT_ERROR | 429 | Rate limit exceeded |
| PRIME_CONFIG_ERROR | 500 | Configuration invalid |
| PRIME_SSM_ERROR | 500 | SSM component error |
| PRIME_MCS_ERROR | 500 | MCS component error |
| PRIME_PREDICTOR_ERROR | 500 | Predictor component error |
| PRIME_SESSION_ERROR | 400 | Invalid session |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| PRIME_HOST | No | 0.0.0.0 | API host |
| PRIME_PORT | No | 8000 | API port |
| PRIME_API_KEY | No | None | API key (enables auth) |
| PRIME_RATE_LIMIT | No | 60 | Requests per minute |
| PRIME_RAGAS_ENABLED | No | true | Enable RAGAS |
| PRIME_RAGAS_MODEL | No | gpt-4.1-mini | RAGAS LLM |
| QDRANT_URL | Yes | - | Qdrant connection URL |
| OPENAI_API_KEY | Yes | - | OpenAI API key |
