"""Unit tests for RAGAS evaluation endpoints.

Tests evaluation API endpoints with mocked RAGAS evaluator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prime.api.app import create_app
from prime.api.models import BatchEvalRequest, EvalRequest

if TYPE_CHECKING:
    pass


@pytest.fixture
def client() -> TestClient:
    """Create test client without evaluator."""
    from prime import PRIMEConfig
    from prime.api.app import MiddlewareConfig, create_app

    app = create_app(
        config=PRIMEConfig.for_testing(),
        middleware_config=MiddlewareConfig.for_testing(),
    )
    # Explicitly set evaluator to None
    app.state.evaluator = None
    return TestClient(app)


@pytest.fixture
def mock_evaluator() -> MagicMock:
    """Create mock RAGASEvaluator."""
    from prime.evaluation import EvaluationResult

    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        faithfulness=0.85,
        context_precision=0.90,
        answer_relevancy=0.88,
        context_recall=0.92,
    )
    return evaluator


@pytest.fixture
def app_with_evaluator(mock_evaluator: MagicMock) -> FastAPI:
    """Create app with mock evaluator."""
    from prime import PRIMEConfig
    from prime.api.app import MiddlewareConfig

    app = create_app(
        config=PRIMEConfig.for_testing(),
        middleware_config=MiddlewareConfig.for_testing(),
    )
    app.state.evaluator = mock_evaluator
    return app


@pytest.fixture
def client_with_evaluator(app_with_evaluator: FastAPI) -> TestClient:
    """Create test client with evaluator."""
    return TestClient(app_with_evaluator)


class TestEvalEndpoint:
    """Tests for POST /diagnostics/eval endpoint."""

    def test_evaluate_rag_success(
        self,
        client_with_evaluator: TestClient,
        mock_evaluator: MagicMock,
    ) -> None:
        """Test successful RAG evaluation."""
        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "contexts": ["Python is a high-level programming language."],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["faithfulness"] == 0.85
        assert data["context_precision"] == 0.90
        assert data["answer_relevancy"] == 0.88
        assert data["context_recall"] == 0.92

        mock_evaluator.evaluate.assert_called_once_with(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
            ground_truth=None,
        )

    def test_evaluate_rag_with_ground_truth(
        self,
        client_with_evaluator: TestClient,
        mock_evaluator: MagicMock,
    ) -> None:
        """Test evaluation with ground truth provided."""
        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "contexts": ["Python is a high-level programming language."],
                "ground_truth": "Python is a versatile programming language.",
            },
        )

        assert response.status_code == 200
        mock_evaluator.evaluate.assert_called_once_with(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
            ground_truth="Python is a versatile programming language.",
        )

    def test_evaluate_rag_missing_question(
        self,
        client_with_evaluator: TestClient,
    ) -> None:
        """Test evaluation fails without question."""
        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval",
            json={
                "answer": "Python is a programming language.",
                "contexts": ["Python is a high-level programming language."],
            },
        )

        assert response.status_code == 422

    def test_evaluate_rag_empty_contexts(
        self,
        client_with_evaluator: TestClient,
    ) -> None:
        """Test evaluation fails with empty contexts."""
        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "contexts": [],
            },
        )

        assert response.status_code == 422

    def test_evaluate_rag_multiple_contexts(
        self,
        client_with_evaluator: TestClient,
        mock_evaluator: MagicMock,
    ) -> None:
        """Test evaluation with multiple contexts."""
        contexts = [
            "Python is a high-level programming language.",
            "Python was created by Guido van Rossum.",
            "Python supports multiple programming paradigms.",
        ]

        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "contexts": contexts,
            },
        )

        assert response.status_code == 200
        mock_evaluator.evaluate.assert_called_once()
        call_args = mock_evaluator.evaluate.call_args
        assert call_args.kwargs["contexts"] == contexts


class TestBatchEvalEndpoint:
    """Tests for POST /diagnostics/eval/batch endpoint."""

    def test_batch_evaluate_success(
        self,
        client_with_evaluator: TestClient,
        mock_evaluator: MagicMock,
    ) -> None:
        """Test successful batch evaluation."""
        from prime.evaluation import BatchEvaluationResult, EvaluationResult

        # Configure mock for batch evaluation
        mock_evaluator.evaluate_batch.return_value = BatchEvaluationResult(
            results=(
                EvaluationResult(
                    faithfulness=0.85,
                    context_precision=0.90,
                    answer_relevancy=0.88,
                    context_recall=0.92,
                ),
                EvaluationResult(
                    faithfulness=0.80,
                    context_precision=0.85,
                    answer_relevancy=0.82,
                    context_recall=0.88,
                ),
            ),
            avg_faithfulness=0.825,
            avg_context_precision=0.875,
            avg_answer_relevancy=0.85,
            avg_context_recall=0.90,
        )

        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval/batch",
            json={
                "samples": [
                    {
                        "question": "What is Python?",
                        "answer": "Python is a programming language.",
                        "contexts": ["Python is a high-level language."],
                    },
                    {
                        "question": "What is Java?",
                        "answer": "Java is a programming language.",
                        "contexts": ["Java is an object-oriented language."],
                    },
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["avg_faithfulness"] == 0.825
        assert data["avg_context_precision"] == 0.875
        assert data["avg_answer_relevancy"] == 0.85
        assert data["avg_context_recall"] == 0.90

    def test_batch_evaluate_empty_samples(
        self,
        client_with_evaluator: TestClient,
    ) -> None:
        """Test batch evaluation fails with empty samples."""
        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval/batch",
            json={"samples": []},
        )

        assert response.status_code == 422

    def test_batch_evaluate_single_sample(
        self,
        client_with_evaluator: TestClient,
        mock_evaluator: MagicMock,
    ) -> None:
        """Test batch evaluation with single sample."""
        from prime.evaluation import BatchEvaluationResult, EvaluationResult

        mock_evaluator.evaluate_batch.return_value = BatchEvaluationResult(
            results=(
                EvaluationResult(
                    faithfulness=0.85,
                    context_precision=0.90,
                    answer_relevancy=0.88,
                    context_recall=0.92,
                ),
            ),
            avg_faithfulness=0.85,
            avg_context_precision=0.90,
            avg_answer_relevancy=0.88,
            avg_context_recall=0.92,
        )

        response = client_with_evaluator.post(
            "/api/v1/diagnostics/eval/batch",
            json={
                "samples": [
                    {
                        "question": "What is Python?",
                        "answer": "Python is a programming language.",
                        "contexts": ["Python is a high-level language."],
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1


class TestEvaluatorNotInitialized:
    """Tests when evaluator is not initialized."""

    def test_eval_without_evaluator(self, client: TestClient) -> None:
        """Test evaluation fails when evaluator not initialized."""
        response = client.post(
            "/api/v1/diagnostics/eval",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "contexts": ["Python is a high-level programming language."],
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "evaluator not initialized" in data["detail"].lower()

    def test_batch_eval_without_evaluator(self, client: TestClient) -> None:
        """Test batch evaluation fails when evaluator not initialized."""
        response = client.post(
            "/api/v1/diagnostics/eval/batch",
            json={
                "samples": [
                    {
                        "question": "What is Python?",
                        "answer": "Python is a programming language.",
                        "contexts": ["Python is a high-level language."],
                    }
                ]
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "evaluator not initialized" in data["detail"].lower()


class TestEvalRequestModel:
    """Tests for EvalRequest model validation."""

    def test_eval_request_valid(self) -> None:
        """Test valid EvalRequest creation."""
        request = EvalRequest(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Context 1", "Context 2"],
        )
        assert request.question == "What is Python?"
        assert len(request.contexts) == 2
        assert request.ground_truth is None

    def test_eval_request_with_ground_truth(self) -> None:
        """Test EvalRequest with ground truth."""
        request = EvalRequest(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Context 1"],
            ground_truth="Expected answer",
        )
        assert request.ground_truth == "Expected answer"


class TestBatchEvalRequestModel:
    """Tests for BatchEvalRequest model validation."""

    def test_batch_request_valid(self) -> None:
        """Test valid BatchEvalRequest creation."""
        request = BatchEvalRequest(
            samples=[
                EvalRequest(
                    question="Q1",
                    answer="A1",
                    contexts=["C1"],
                ),
                EvalRequest(
                    question="Q2",
                    answer="A2",
                    contexts=["C2"],
                ),
            ]
        )
        assert len(request.samples) == 2


class TestEvaluationTypes:
    """Tests for evaluation type classes."""

    def test_eval_sample_creation(self) -> None:
        """Test EvalSample dataclass."""
        from prime.evaluation import EvalSample

        sample = EvalSample(
            question="What is Python?",
            answer="A programming language.",
            contexts=["Python is versatile."],
            ground_truth="Python is a programming language.",
        )

        assert sample.question == "What is Python?"
        assert sample.answer == "A programming language."
        assert sample.contexts == ["Python is versatile."]
        assert sample.ground_truth == "Python is a programming language."

    def test_eval_sample_immutable(self) -> None:
        """Test EvalSample is frozen."""
        from prime.evaluation import EvalSample

        sample = EvalSample(
            question="Q",
            answer="A",
            contexts=["C"],
        )

        with pytest.raises(AttributeError):
            sample.question = "New Q"  # type: ignore[misc]

    def test_evaluation_result_creation(self) -> None:
        """Test EvaluationResult dataclass."""
        from prime.evaluation import EvaluationResult

        result = EvaluationResult(
            faithfulness=0.85,
            context_precision=0.90,
            answer_relevancy=0.88,
            context_recall=0.92,
        )

        assert result.faithfulness == 0.85
        assert result.context_precision == 0.90
        assert result.answer_relevancy == 0.88
        assert result.context_recall == 0.92

    def test_evaluation_result_without_recall(self) -> None:
        """Test EvaluationResult without context_recall."""
        from prime.evaluation import EvaluationResult

        result = EvaluationResult(
            faithfulness=0.85,
            context_precision=0.90,
            answer_relevancy=0.88,
        )

        assert result.context_recall is None

    def test_batch_evaluation_result_from_results(self) -> None:
        """Test BatchEvaluationResult.from_results factory."""
        from prime.evaluation import BatchEvaluationResult, EvaluationResult

        results = [
            EvaluationResult(
                faithfulness=0.80,
                context_precision=0.90,
                answer_relevancy=0.85,
                context_recall=0.88,
            ),
            EvaluationResult(
                faithfulness=0.90,
                context_precision=0.80,
                answer_relevancy=0.75,
                context_recall=0.92,
            ),
        ]

        batch = BatchEvaluationResult.from_results(results)

        assert len(batch.results) == 2
        assert batch.avg_faithfulness == pytest.approx(0.85)
        assert batch.avg_context_precision == pytest.approx(0.85)
        assert batch.avg_answer_relevancy == pytest.approx(0.80)
        assert batch.avg_context_recall == pytest.approx(0.90)

    def test_batch_evaluation_result_without_recall(self) -> None:
        """Test BatchEvaluationResult without context_recall values."""
        from prime.evaluation import BatchEvaluationResult, EvaluationResult

        results = [
            EvaluationResult(
                faithfulness=0.80,
                context_precision=0.90,
                answer_relevancy=0.85,
            ),
            EvaluationResult(
                faithfulness=0.90,
                context_precision=0.80,
                answer_relevancy=0.75,
            ),
        ]

        batch = BatchEvaluationResult.from_results(results)

        assert batch.avg_context_recall is None
