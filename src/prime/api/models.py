"""API request and response models.

Pydantic models for FastAPI endpoint validation and serialization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Request for processing a conversation turn."""

    input: str = Field(min_length=1, max_length=8192)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str | None = None
    force_retrieval: bool = False
    k: int = Field(default=5, ge=1, le=20)


class MemoryResult(BaseModel):
    """Memory result in API response."""

    memory_id: str
    content: str
    cluster_id: int
    similarity: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = 0.0


class ProcessResponse(BaseModel):
    """Response from process_turn."""

    retrieved_memories: list[MemoryResult]
    boundary_crossed: bool
    variance: float
    smoothed_variance: float
    action: str
    session_id: str
    turn_number: int
    latency_ms: float


class MemoryWriteRequest(BaseModel):
    """Request to write memory."""

    content: str = Field(min_length=1, max_length=50000)
    metadata: dict[str, str | int | float | bool] | None = None
    user_id: str | None = None
    session_id: str | None = None


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


class MemorySearchResponse(BaseModel):
    """Response from memory search."""

    results: list[MemoryResult]
    query_embedding: list[float] | None = None


class ComponentStatusResponse(BaseModel):
    """Status of individual component."""

    name: str
    status: str
    latency_p50_ms: float
    error_rate: float


class DiagnosticsResponse(BaseModel):
    """System diagnostics response."""

    status: str
    version: str
    uptime_seconds: float
    components: dict[str, ComponentStatusResponse]
    metrics: dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class ClusterInfoResponse(BaseModel):
    """Cluster information response."""

    cluster_id: int
    size: int
    is_consolidated: bool
    prototype_norm: float
    creation_timestamp: float
    last_access_timestamp: float
    access_count: int
    representative_content: str | None = None


class ClusterListResponse(BaseModel):
    """List of clusters response."""

    clusters: list[ClusterInfoResponse]
    total_count: int


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    variance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    consolidation_threshold: int | None = Field(default=None, ge=1, le=100)


class ConfigUpdateResponse(BaseModel):
    """Response from config update."""

    updated: bool
    changes: dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    error_code: str
    detail: str | None = None
    request_id: str


class EvalRequest(BaseModel):
    """Request for RAG evaluation."""

    question: str = Field(min_length=1, max_length=8192)
    answer: str = Field(min_length=1, max_length=50000)
    contexts: list[str] = Field(min_length=1)
    ground_truth: str | None = None


class EvalResponse(BaseModel):
    """Response from RAG evaluation."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchEvalRequest(BaseModel):
    """Request for batch RAG evaluation."""

    samples: list[EvalRequest] = Field(min_length=1, max_length=100)


class BatchEvalResponse(BaseModel):
    """Response from batch RAG evaluation."""

    results: list[EvalResponse]
    avg_faithfulness: float = Field(ge=0.0, le=1.0)
    avg_context_precision: float = Field(ge=0.0, le=1.0)
    avg_answer_relevancy: float = Field(ge=0.0, le=1.0)
    avg_context_recall: float | None = Field(default=None, ge=0.0, le=1.0)
