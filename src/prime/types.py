"""PRIME API type definitions.

Shared type definitions for the PRIME orchestration layer including
action states, memory results, and diagnostic structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Re-export ActionState from SSM module to avoid duplication
from prime.ssm.ssm_types import ActionState

__all__ = [
    "ActionState",
    "ComponentStatus",
    "MemoryReadResult",
    "MemoryWriteResult",
    "PRIMEDiagnostics",
    "PRIMEResponse",
]


def _empty_metadata() -> dict[str, Any]:
    """Create empty metadata dict for dataclass defaults."""
    return {}


@dataclass(frozen=True, slots=True)
class MemoryReadResult:
    """Memory retrieval result from MCS search.

    Represents a single memory retrieved from the Memory Cluster Store
    with similarity score and metadata.

    Attributes:
        memory_id: Unique identifier for the memory.
        content: Text content of the memory.
        cluster_id: ID of the cluster containing this memory.
        similarity: Cosine similarity to the query vector.
        metadata: Additional key-value metadata.
        created_at: Unix timestamp of memory creation.
    """

    memory_id: str
    content: str
    cluster_id: int
    similarity: float
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Memory write result from MCS storage.

    Represents the outcome of storing a memory in MCS including
    cluster assignment and consolidation status.

    Attributes:
        memory_id: Unique identifier assigned to the memory.
        cluster_id: ID of the cluster containing this memory.
        is_new_cluster: Whether a new cluster was created.
        consolidated: Whether cluster consolidation occurred.
    """

    memory_id: str
    cluster_id: int
    is_new_cluster: bool
    consolidated: bool


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    """Individual component health status.

    Reports the operational status of a single PRIME component
    including latency and error metrics.

    Attributes:
        name: Component name (e.g., "ssm", "mcs", "predictor").
        status: Health status ("healthy", "degraded", "unhealthy").
        latency_p50_ms: 50th percentile latency in milliseconds.
        error_rate: Error rate as a decimal (0.01 = 1%).
    """

    name: str
    status: str
    latency_p50_ms: float
    error_rate: float


@dataclass(frozen=True, slots=True)
class PRIMEResponse:
    """Response from PRIME.process_turn().

    Contains retrieved memories, boundary detection results,
    and session state information.

    Attributes:
        retrieved_memories: List of memories retrieved from MCS.
        boundary_crossed: Whether semantic boundary was detected.
        variance: Current variance value from SSM.
        smoothed_variance: EMA-smoothed variance value.
        action: Action state determined by SSM.
        session_id: Session identifier for this conversation.
        turn_number: Turn number within the session.
        latency_ms: Processing latency in milliseconds.
    """

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
    """System diagnostics for PRIME.

    Provides comprehensive health and performance information
    for all PRIME components.

    Attributes:
        status: Overall system status ("healthy", "degraded", "unhealthy").
        version: PRIME API version string.
        uptime_seconds: Time since system startup in seconds.
        components: Health status per component.
        metrics: Key performance metrics.
    """

    status: str
    version: str
    uptime_seconds: float
    components: dict[str, ComponentStatus]
    metrics: dict[str, float]
