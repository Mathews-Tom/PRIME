"""Memory endpoints.

POST /memory/write - Write external knowledge to memory
POST /memory/search - Direct memory search
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from prime.api.dependencies import get_prime
from prime.api.models import (
    MemoryResult,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
)

if TYPE_CHECKING:
    from prime import PRIME

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/write", response_model=MemoryWriteResponse)
def write_memory(
    request: MemoryWriteRequest,
    prime: PRIME = Depends(get_prime),
) -> MemoryWriteResponse:
    """Write external knowledge to memory.

    Use this for adding knowledge that doesn't come from conversation,
    such as documents, articles, or pre-existing knowledge bases.

    Args:
        request: MemoryWriteRequest with content and metadata.
        prime: PRIME instance (injected).

    Returns:
        MemoryWriteResponse with memory_id and cluster assignment.
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
def search_memory(
    request: MemorySearchRequest,
    prime: PRIME = Depends(get_prime),
) -> MemorySearchResponse:
    """Search memory directly.

    Use this for explicit search operations where you want to bypass
    the predictive retrieval mechanism and directly query memories.

    Args:
        request: MemorySearchRequest with query and options.
        prime: PRIME instance (injected).

    Returns:
        MemorySearchResponse with matching memories.
    """
    results = prime.search_memory(
        query=request.query,
        k=request.k,
        min_similarity=request.min_similarity,
        session_id=request.session_id,
    )

    return MemorySearchResponse(
        results=[
            MemoryResult(
                memory_id=mem.memory_id,
                content=mem.content,
                cluster_id=mem.cluster_id,
                similarity=mem.similarity,
                metadata=mem.metadata or {},
                created_at=mem.created_at,
            )
            for mem in results
        ],
        query_embedding=None,  # Optionally expose embedding
    )
