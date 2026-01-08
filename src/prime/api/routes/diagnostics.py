"""Diagnostics and health endpoints.

GET /diagnostics - System diagnostics with component health
GET /health - Simple health check
POST /diagnostics/eval - RAG evaluation using RAGAS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from prime.api.dependencies import get_evaluator, get_prime
from prime.api.models import (
    BatchEvalRequest,
    BatchEvalResponse,
    ComponentStatusResponse,
    DiagnosticsResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
)

if TYPE_CHECKING:
    from prime import PRIME
    from prime.evaluation import RAGASEvaluator

router = APIRouter(tags=["diagnostics"])


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics(
    prime: PRIME = Depends(get_prime),
) -> DiagnosticsResponse:
    """Get comprehensive system diagnostics.

    Returns health status, performance metrics, and component
    information for monitoring and debugging.

    Args:
        prime: PRIME instance (injected).

    Returns:
        DiagnosticsResponse with status, uptime, components, and metrics.
    """
    diag = prime.get_diagnostics()

    return DiagnosticsResponse(
        status=diag.status,
        version=diag.version,
        uptime_seconds=diag.uptime_seconds,
        components={
            name: ComponentStatusResponse(
                name=comp.name,
                status=comp.status,
                latency_p50_ms=comp.latency_p50_ms,
                error_rate=comp.error_rate,
            )
            for name, comp in diag.components.items()
        },
        metrics=diag.metrics,
    )


@router.get("/health", response_model=HealthResponse)
def health_check(
    prime: PRIME = Depends(get_prime),
) -> HealthResponse:
    """Simple health check endpoint.

    Returns basic health status for load balancer probes.

    Args:
        prime: PRIME instance (injected).

    Returns:
        HealthResponse with status and version.
    """
    diag = prime.get_diagnostics()

    return HealthResponse(
        status=diag.status,
        version=diag.version,
    )


@router.post("/diagnostics/eval", response_model=EvalResponse)
def evaluate_rag(
    request: EvalRequest,
    evaluator: RAGASEvaluator = Depends(get_evaluator),
) -> EvalResponse:
    """Evaluate RAG quality using RAGAS metrics.

    Computes faithfulness, context precision, and answer relevancy
    scores for a single question-answer-context tuple.

    Args:
        request: Evaluation request with question, answer, contexts.
        evaluator: RAGAS evaluator instance (injected).

    Returns:
        EvalResponse with metric scores.
    """
    result = evaluator.evaluate(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        ground_truth=request.ground_truth,
    )

    return EvalResponse(
        faithfulness=result.faithfulness,
        context_precision=result.context_precision,
        answer_relevancy=result.answer_relevancy,
        context_recall=result.context_recall,
    )


@router.post("/diagnostics/eval/batch", response_model=BatchEvalResponse)
def evaluate_rag_batch(
    request: BatchEvalRequest,
    evaluator: RAGASEvaluator = Depends(get_evaluator),
) -> BatchEvalResponse:
    """Evaluate multiple RAG samples in batch.

    Computes RAGAS metrics for multiple samples efficiently.

    Args:
        request: Batch evaluation request with list of samples.
        evaluator: RAGAS evaluator instance (injected).

    Returns:
        BatchEvalResponse with individual and aggregate scores.
    """
    from prime.evaluation import EvalSample

    samples = [
        EvalSample(
            question=s.question,
            answer=s.answer,
            contexts=s.contexts,
            ground_truth=s.ground_truth,
        )
        for s in request.samples
    ]

    result = evaluator.evaluate_batch(samples)

    return BatchEvalResponse(
        results=[
            EvalResponse(
                faithfulness=r.faithfulness,
                context_precision=r.context_precision,
                answer_relevancy=r.answer_relevancy,
                context_recall=r.context_recall,
            )
            for r in result.results
        ],
        avg_faithfulness=result.avg_faithfulness,
        avg_context_precision=result.avg_context_precision,
        avg_answer_relevancy=result.avg_answer_relevancy,
        avg_context_recall=result.avg_context_recall,
    )
