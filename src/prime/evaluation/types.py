"""Evaluation type definitions.

Data classes for RAGAS evaluation results and samples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSample:
    """A single sample for evaluation.

    Attributes:
        question: The user question/query.
        answer: The generated answer.
        contexts: Retrieved context passages.
        ground_truth: Optional reference answer for recall.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Result from RAGAS evaluation.

    Attributes:
        faithfulness: Score for answer grounded in context (0-1).
        context_precision: Score for relevant context ranking (0-1).
        answer_relevancy: Score for answer addressing question (0-1).
        context_recall: Score for ground truth in context (0-1), requires ground_truth.
    """

    faithfulness: float
    context_precision: float
    answer_relevancy: float
    context_recall: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all metric scores.
        """
        return {
            "faithfulness": self.faithfulness,
            "context_precision": self.context_precision,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
        }


@dataclass(frozen=True)
class BatchEvaluationResult:
    """Result from batch RAGAS evaluation.

    Attributes:
        results: Individual evaluation results.
        avg_faithfulness: Average faithfulness across samples.
        avg_context_precision: Average context precision across samples.
        avg_answer_relevancy: Average answer relevancy across samples.
        avg_context_recall: Average context recall (if applicable).
    """

    results: tuple[EvaluationResult, ...]
    avg_faithfulness: float
    avg_context_precision: float
    avg_answer_relevancy: float
    avg_context_recall: float | None = None

    @classmethod
    def from_results(cls, results: list[EvaluationResult]) -> BatchEvaluationResult:
        """Create batch result with computed averages.

        Args:
            results: List of individual evaluation results.

        Returns:
            BatchEvaluationResult with computed averages.
        """
        n = len(results)
        if n == 0:
            return cls(
                results=(),
                avg_faithfulness=0.0,
                avg_context_precision=0.0,
                avg_answer_relevancy=0.0,
                avg_context_recall=None,
            )

        avg_faith = sum(r.faithfulness for r in results) / n
        avg_precision = sum(r.context_precision for r in results) / n
        avg_relevancy = sum(r.answer_relevancy for r in results) / n

        # Compute recall average only if any result has it
        recall_values = [r.context_recall for r in results if r.context_recall is not None]
        avg_recall = sum(recall_values) / len(recall_values) if recall_values else None

        return cls(
            results=tuple(results),
            avg_faithfulness=avg_faith,
            avg_context_precision=avg_precision,
            avg_answer_relevancy=avg_relevancy,
            avg_context_recall=avg_recall,
        )
