"""RAGAS evaluation integration.

Provides RAG quality evaluation using RAGAS metrics:
- Faithfulness: Answer grounded in context
- Context Precision: Relevant context ranking
- Answer Relevancy: Answer addresses question
- Context Recall: Ground truth in context (optional)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from prime.evaluation.types import (
    BatchEvaluationResult,
    EvalSample,
    EvaluationResult,
)
from prime.exceptions import ComponentError, ConfigurationError

if TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics.base import Metric

    from prime.config import RAGASConfig


class RAGASEvaluator:
    """Evaluator for RAG quality using RAGAS metrics.

    Provides reference-free evaluation of RAG system outputs using
    LLM-based metrics for faithfulness, relevancy, and precision.

    Attributes:
        config: RAGAS configuration.
    """

    def __init__(
        self,
        config: RAGASConfig,
        llm: BaseRagasLLM | None = None,
    ) -> None:
        """Initialize the RAGAS evaluator.

        Args:
            config: RAGAS configuration.
            llm: Optional pre-configured LLM wrapper. If None, creates one
                using config.llm_model.

        Raises:
            ConfigurationError: If LLM cannot be initialized.
        """
        self.config = config
        self._llm = llm or self._create_llm()
        self._metrics = self._create_metrics()

    def _create_llm(self) -> BaseRagasLLM:
        """Create LLM wrapper for RAGAS evaluation.

        Returns:
            Configured RAGAS LLM wrapper.

        Raises:
            ConfigurationError: If API key is missing or LLM fails to initialize.
        """
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY environment variable required for RAGAS evaluation"
            )

        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=api_key,
                timeout=self.config.timeout_seconds,
            )
            return LangchainLLMWrapper(llm)
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import LangChain OpenAI: {e}. "
                "Install with: uv add langchain-openai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to create LLM: {e}") from e

    def _create_metrics(self) -> list[Metric]:
        """Create RAGAS metric instances.

        Returns:
            List of configured RAGAS metrics.
        """
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

        metrics: list[Metric] = [
            Faithfulness(llm=self._llm),
            ContextPrecision(llm=self._llm),
            AnswerRelevancy(llm=self._llm),
        ]

        return metrics

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single RAG sample.

        Args:
            question: The user question/query.
            answer: The generated answer.
            contexts: Retrieved context passages.
            ground_truth: Optional reference answer for recall metric.

        Returns:
            EvaluationResult with metric scores.

        Raises:
            ComponentError: If evaluation fails.
        """
        if not self.config.enabled:
            raise ComponentError("RAGAS evaluation is disabled")

        try:
            from ragas import EvaluationDataset, evaluate

            # Build sample dict
            sample_dict: dict[str, str | list[str]] = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            }
            if ground_truth is not None:
                sample_dict["reference"] = ground_truth

            dataset = EvaluationDataset.from_list([sample_dict])

            result = evaluate(
                dataset=dataset,
                metrics=self._metrics,
            )

            # Extract scores from result DataFrame
            df = result.to_pandas()

            return EvaluationResult(
                faithfulness=float(df["faithfulness"].iloc[0]),
                context_precision=float(df["context_precision"].iloc[0]),
                answer_relevancy=float(df["answer_relevancy"].iloc[0]),
                context_recall=None,  # Would require context_recall metric
            )
        except Exception as e:
            raise ComponentError(f"RAGAS evaluation failed: {e}") from e

    def evaluate_batch(
        self,
        samples: list[EvalSample],
    ) -> BatchEvaluationResult:
        """Evaluate multiple RAG samples efficiently.

        Args:
            samples: List of evaluation samples.

        Returns:
            BatchEvaluationResult with individual and aggregate scores.

        Raises:
            ComponentError: If evaluation fails.
        """
        if not self.config.enabled:
            raise ComponentError("RAGAS evaluation is disabled")

        if not samples:
            return BatchEvaluationResult.from_results([])

        try:
            from ragas import EvaluationDataset, evaluate

            # Build dataset
            sample_dicts: list[dict[str, str | list[str]]] = []
            for sample in samples:
                d: dict[str, str | list[str]] = {
                    "user_input": sample.question,
                    "response": sample.answer,
                    "retrieved_contexts": sample.contexts,
                }
                if sample.ground_truth is not None:
                    d["reference"] = sample.ground_truth
                sample_dicts.append(d)

            dataset = EvaluationDataset.from_list(sample_dicts)

            result = evaluate(
                dataset=dataset,
                metrics=self._metrics,
            )

            # Extract scores from result DataFrame
            df = result.to_pandas()

            results: list[EvaluationResult] = []
            for i in range(len(samples)):
                results.append(
                    EvaluationResult(
                        faithfulness=float(df["faithfulness"].iloc[i]),
                        context_precision=float(df["context_precision"].iloc[i]),
                        answer_relevancy=float(df["answer_relevancy"].iloc[i]),
                        context_recall=None,
                    )
                )

            return BatchEvaluationResult.from_results(results)
        except Exception as e:
            raise ComponentError(f"RAGAS batch evaluation failed: {e}") from e
