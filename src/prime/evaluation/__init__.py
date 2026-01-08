"""Evaluation module for PRIME.

Provides RAGAS-based RAG quality evaluation.

Example:
    >>> from prime.evaluation import RAGASEvaluator
    >>> from prime.config import RAGASConfig
    >>> evaluator = RAGASEvaluator(RAGASConfig())
    >>> result = evaluator.evaluate(
    ...     question="What is Python?",
    ...     answer="Python is a programming language.",
    ...     contexts=["Python is a high-level programming language."],
    ... )
"""

from __future__ import annotations

from prime.evaluation.ragas import RAGASEvaluator
from prime.evaluation.types import (
    BatchEvaluationResult,
    EvalSample,
    EvaluationResult,
)

__all__ = [
    "BatchEvaluationResult",
    "EvalSample",
    "EvaluationResult",
    "RAGASEvaluator",
]
