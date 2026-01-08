"""Process conversation turn endpoint.

POST /process - Main endpoint for processing conversation turns
through the PRIME predictive retrieval pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from prime.api.dependencies import get_prime
from prime.api.models import MemoryResult, ProcessRequest, ProcessResponse

if TYPE_CHECKING:
    from prime import PRIME

router = APIRouter(tags=["process"])


@router.post("/process", response_model=ProcessResponse)
def process_turn(
    request: ProcessRequest,
    prime: PRIME = Depends(get_prime),
) -> ProcessResponse:
    """Process a conversation turn through the PRIME pipeline.

    This is the main entry point for predictive RAG processing. It:
    1. Updates SSM state and checks for semantic boundary crossing
    2. If boundary crossed or force_retrieval: predicts target embedding
    3. Searches MCS with predicted embedding for relevant memories
    4. Returns retrieved memories with boundary detection info

    Args:
        request: ProcessRequest with input text and options.
        prime: PRIME instance (injected).

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
            MemoryResult(
                memory_id=mem.memory_id,
                content=mem.content,
                cluster_id=mem.cluster_id,
                similarity=mem.similarity,
                metadata=mem.metadata or {},
                created_at=mem.created_at,
            )
            for mem in result.retrieved_memories
        ],
        boundary_crossed=result.boundary_crossed,
        variance=result.variance,
        smoothed_variance=result.smoothed_variance,
        action=result.action.value,
        session_id=result.session_id,
        turn_number=result.turn_number,
        latency_ms=result.latency_ms,
    )
