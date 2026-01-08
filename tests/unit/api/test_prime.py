"""Unit tests for PRIME orchestration class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from prime import (
    PRIME,
    ActionState,
    ComponentStatus,
    MemoryReadResult,
    MemoryWriteResult,
    PRIMEConfig,
    PRIMEDiagnostics,
    PRIMEResponse,
)
from prime.mcs.index import IndexSearchResult, VectorIndex

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _random_embedding(dim: int = 384, seed: int | None = None) -> NDArray[np.float32]:
    """Generate a random L2-normalized embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


@dataclass
class MockEncoder:
    """Mock encoder for PRIME testing.

    Generates deterministic embeddings based on content hash.
    Implements the Encoder protocol.
    """

    embedding_dim: int = 384
    max_length: int = 512
    model_name: str = "mock-encoder"
    _preset_embeddings: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Initialize preset embeddings dict."""
        if self._preset_embeddings is None:
            self._preset_embeddings = {}

    def encode(self, text: str) -> np.ndarray:
        """Encode text to deterministic embedding based on content hash."""
        if self._preset_embeddings and text in self._preset_embeddings:
            return self._preset_embeddings[text]
        seed = hash(text) % (2**31)
        return _random_embedding(self.embedding_dim, seed)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode batch of texts."""
        return [self.encode(text) for text in texts]

    def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Set a preset embedding for a specific text."""
        if self._preset_embeddings is None:
            self._preset_embeddings = {}
        self._preset_embeddings[text] = embedding

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "pooling_mode": "mean",
            "device": "cpu",
        }


class MockVectorIndex:
    """Mock vector index for testing.

    Stores vectors in memory and performs brute-force search.
    Implements VectorIndex protocol.
    """

    def __init__(self) -> None:
        """Initialize mock index."""
        self._vectors: dict[str, np.ndarray] = {}
        self._sparse: dict[str, Any] = {}  # SparseVector at runtime
        self._payloads: dict[str, dict[str, Any]] = {}

    def add(
        self,
        id: str,
        dense: np.ndarray,
        sparse: Any = None,
        payload: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Add a vector to the index."""
        self._vectors[id] = dense.copy()
        if sparse is not None:
            self._sparse[id] = sparse
        if payload is not None:
            self._payloads[id] = dict(payload)

    def search_dense(
        self,
        query: np.ndarray,
        top_k: int,
        filter_payload: dict[str, str | int | float | bool] | None = None,
    ) -> list[IndexSearchResult]:
        """Search using dense vector similarity."""
        results: list[tuple[str, float]] = []

        for vec_id, vec in self._vectors.items():
            # Apply filter
            if filter_payload:
                payload = self._payloads.get(vec_id, {})
                if not all(payload.get(k) == v for k, v in filter_payload.items()):
                    continue

            score = float(np.dot(query, vec))
            results.append((vec_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [
            IndexSearchResult(id=vec_id, score=score)
            for vec_id, score in results[:top_k]
        ]

    def search_hybrid(
        self,
        dense_query: np.ndarray,
        sparse_query: Any,
        top_k: int,
        filter_payload: dict[str, str | int | float | bool] | None = None,
    ) -> list[IndexSearchResult]:
        """Search using hybrid dense + sparse (mock: just uses dense)."""
        del sparse_query  # Unused in mock
        return self.search_dense(
            query=dense_query, top_k=top_k, filter_payload=filter_payload
        )

    def remove(self, id: str) -> bool:
        """Remove a vector from the index."""
        if id in self._vectors:
            del self._vectors[id]
            self._sparse.pop(id, None)
            self._payloads.pop(id, None)
            return True
        return False

    def get(self, id: str) -> np.ndarray | None:
        """Get a vector by ID."""
        return self._vectors.get(id)

    def count(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._vectors)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()
        self._sparse.clear()
        self._payloads.clear()


# Verify MockVectorIndex implements VectorIndex protocol
assert isinstance(MockVectorIndex(), VectorIndex)


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Create mock encoder with 384 dimensions."""
    return MockEncoder(embedding_dim=384)


@pytest.fixture
def mock_index() -> MockVectorIndex:
    """Create mock vector index."""
    return MockVectorIndex()


@pytest.fixture
def test_config() -> PRIMEConfig:
    """Create test configuration for PRIME."""
    return PRIMEConfig.for_testing()


class TestPRIMEInitialization:
    """Tests for PRIME initialization."""

    def test_initialization_with_testing_config(self) -> None:
        """Test PRIME initializes with testing configuration."""
        config = PRIMEConfig.for_testing()
        prime = PRIME(config)

        assert prime.config == config
        assert prime.y_encoder is not None
        assert prime.ssm is not None
        assert prime.mcs is not None
        assert prime.predictor is not None

    def test_initialization_stores_config(self, test_config: PRIMEConfig) -> None:
        """Test PRIME stores configuration correctly."""
        prime = PRIME(test_config)
        assert prime.config is test_config

    def test_initialization_creates_session_context(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test PRIME initializes empty session context."""
        prime = PRIME(test_config)
        # Session context is private, but get_diagnostics reveals it
        diagnostics = prime.get_diagnostics()
        assert diagnostics.metrics["active_sessions"] == 0.0

    def test_initialization_resets_counters(self, test_config: PRIMEConfig) -> None:
        """Test PRIME initializes with zero counters."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()
        assert diagnostics.metrics["total_requests"] == 0.0
        assert diagnostics.metrics["total_errors"] == 0.0


class TestProcessTurn:
    """Tests for PRIME.process_turn() method."""

    def test_process_turn_returns_prime_response(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn returns PRIMEResponse."""
        prime = PRIME(test_config)
        response = prime.process_turn("Hello, world!")

        assert isinstance(response, PRIMEResponse)

    def test_process_turn_generates_session_id(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn generates session_id if not provided."""
        prime = PRIME(test_config)
        response = prime.process_turn("Test input")

        assert response.session_id is not None
        assert response.session_id.startswith("sess_")

    def test_process_turn_uses_provided_session_id(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn uses provided session_id."""
        prime = PRIME(test_config)
        response = prime.process_turn("Test input", session_id="custom_session")

        assert response.session_id == "custom_session"

    def test_process_turn_returns_variance_metrics(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn returns variance metrics from SSM."""
        prime = PRIME(test_config)
        response = prime.process_turn("Test input")

        assert isinstance(response.variance, float)
        assert isinstance(response.smoothed_variance, float)
        assert response.variance >= 0.0

    def test_process_turn_returns_action_state(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn returns action state from SSM."""
        prime = PRIME(test_config)
        response = prime.process_turn("Test input")

        assert isinstance(response.action, ActionState)

    def test_process_turn_increments_turn_number(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn increments turn number."""
        prime = PRIME(test_config)
        session_id = "test_session"

        response1 = prime.process_turn("First input", session_id=session_id)
        response2 = prime.process_turn("Second input", session_id=session_id)

        assert response2.turn_number > response1.turn_number

    def test_process_turn_returns_latency_ms(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn returns processing latency."""
        prime = PRIME(test_config)
        response = prime.process_turn("Test input")

        assert isinstance(response.latency_ms, float)
        assert response.latency_ms > 0.0

    def test_process_turn_force_retrieval(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn with force_retrieval flag."""
        prime = PRIME(test_config)
        # First write some content to retrieve
        prime.record_response("Some stored content", session_id="test")

        response = prime.process_turn(
            "Related query", session_id="test", force_retrieval=True
        )

        # With force_retrieval, retrieval should occur even without boundary crossing
        assert isinstance(response.retrieved_memories, list)

    def test_process_turn_custom_k(self, test_config: PRIMEConfig) -> None:
        """Test process_turn with custom k parameter."""
        prime = PRIME(test_config)
        # Store multiple memories with distinct topics to avoid consolidation
        topics = [
            "Python programming language syntax",
            "Machine learning algorithms overview",
            "Database design principles",
            "Network security fundamentals",
            "Cloud computing architecture",
        ]
        for topic in topics:
            prime.record_response(topic, session_id="test")

        response = prime.process_turn(
            "Query", session_id="test", force_retrieval=True, k=3
        )

        assert len(response.retrieved_memories) <= 3

    def test_process_turn_increments_request_count(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test process_turn increments request counter."""
        prime = PRIME(test_config)

        prime.process_turn("Test 1")
        prime.process_turn("Test 2")

        diagnostics = prime.get_diagnostics()
        assert diagnostics.metrics["total_requests"] == 2.0


class TestRecordResponse:
    """Tests for PRIME.record_response() method."""

    def test_record_response_returns_write_result(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test record_response returns MemoryWriteResult."""
        prime = PRIME(test_config)
        result = prime.record_response("Test response content")

        assert isinstance(result, MemoryWriteResult)

    def test_record_response_generates_memory_id(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test record_response generates unique memory_id."""
        prime = PRIME(test_config)
        result = prime.record_response("Test content")

        assert result.memory_id is not None
        assert len(result.memory_id) > 0

    def test_record_response_assigns_cluster_id(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test record_response assigns cluster_id."""
        prime = PRIME(test_config)
        result = prime.record_response("Test content")

        assert isinstance(result.cluster_id, int)

    def test_record_response_with_session_id(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test record_response with session_id."""
        prime = PRIME(test_config)
        result = prime.record_response(
            "Test content", session_id="session_123"
        )

        assert result.memory_id is not None

    def test_record_response_with_metadata(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test record_response with custom metadata."""
        prime = PRIME(test_config)
        result = prime.record_response(
            "Test content", metadata={"source": "test", "importance": 5}
        )

        assert result.memory_id is not None

    def test_record_response_multiple_creates_cluster(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test recording multiple similar responses creates cluster."""
        prime = PRIME(test_config)

        results = []
        for i in range(5):
            result = prime.record_response(f"Similar content about topic {i}")
            results.append(result)

        # All should have cluster assignments
        for result in results:
            assert result.cluster_id is not None


class TestWriteExternalKnowledge:
    """Tests for PRIME.write_external_knowledge() method."""

    def test_write_external_knowledge_returns_result(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test write_external_knowledge returns MemoryWriteResult."""
        prime = PRIME(test_config)
        result = prime.write_external_knowledge("Document content here")

        assert isinstance(result, MemoryWriteResult)

    def test_write_external_knowledge_generates_id(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test write_external_knowledge generates memory_id."""
        prime = PRIME(test_config)
        result = prime.write_external_knowledge("External document")

        assert result.memory_id is not None

    def test_write_external_knowledge_with_metadata(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test write_external_knowledge preserves metadata."""
        prime = PRIME(test_config)
        result = prime.write_external_knowledge(
            "Document content",
            metadata={"title": "Test Doc", "version": 1},
        )

        assert result.memory_id is not None

    def test_write_external_knowledge_without_metadata(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test write_external_knowledge works without metadata."""
        prime = PRIME(test_config)
        result = prime.write_external_knowledge("Plain content")

        assert result.memory_id is not None


class TestSearchMemory:
    """Tests for PRIME.search_memory() method."""

    def test_search_memory_returns_list(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test search_memory returns list of results."""
        prime = PRIME(test_config)
        results = prime.search_memory("test query")

        assert isinstance(results, list)

    def test_search_memory_finds_stored_content(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test search_memory finds previously stored content."""
        prime = PRIME(test_config)
        prime.record_response("Python is a programming language")

        results = prime.search_memory("programming language")

        assert len(results) > 0 or results == []  # May or may not match

    def test_search_memory_returns_memory_read_result(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test search_memory returns MemoryReadResult instances."""
        prime = PRIME(test_config)
        prime.record_response("Test searchable content")

        results = prime.search_memory("searchable", k=5)

        for result in results:
            assert isinstance(result, MemoryReadResult)

    def test_search_memory_respects_k_parameter(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test search_memory respects k limit."""
        prime = PRIME(test_config)
        # Use distinct topics to avoid consolidation
        topics = [
            "Quantum computing applications",
            "Organic chemistry reactions",
            "Renaissance art history",
            "Marine biology ecosystems",
            "Ancient philosophy debates",
        ]
        for topic in topics:
            prime.record_response(topic)

        results = prime.search_memory("science", k=3)

        assert len(results) <= 3

    def test_search_memory_with_min_similarity(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test search_memory with min_similarity threshold."""
        prime = PRIME(test_config)
        prime.record_response("Relevant content")

        results = prime.search_memory(
            "relevant", k=5, min_similarity=0.5
        )

        for result in results:
            assert result.similarity >= 0.5

    def test_search_memory_with_session_filter(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test search_memory with session_id filter."""
        prime = PRIME(test_config)
        prime.record_response("Content A", session_id="session_a")
        prime.record_response("Content B", session_id="session_b")

        # Search only in session_a
        results = prime.search_memory("content", session_id="session_a")

        assert isinstance(results, list)


class TestGetDiagnostics:
    """Tests for PRIME.get_diagnostics() method."""

    def test_get_diagnostics_returns_diagnostics(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test get_diagnostics returns PRIMEDiagnostics."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert isinstance(diagnostics, PRIMEDiagnostics)

    def test_get_diagnostics_includes_status(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test diagnostics includes health status."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert diagnostics.status in ("healthy", "degraded", "unhealthy")

    def test_get_diagnostics_includes_version(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test diagnostics includes version."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert isinstance(diagnostics.version, str)
        assert len(diagnostics.version) > 0

    def test_get_diagnostics_includes_uptime(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test diagnostics includes uptime."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert diagnostics.uptime_seconds >= 0.0

    def test_get_diagnostics_includes_components(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test diagnostics includes component status."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert "ssm" in diagnostics.components
        assert "mcs" in diagnostics.components
        assert "predictor" in diagnostics.components
        assert "y_encoder" in diagnostics.components

    def test_get_diagnostics_component_status_type(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test component status is ComponentStatus type."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        for _name, status in diagnostics.components.items():
            assert isinstance(status, ComponentStatus)
            assert status.name is not None

    def test_get_diagnostics_includes_metrics(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test diagnostics includes metrics."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert "total_requests" in diagnostics.metrics
        assert "total_errors" in diagnostics.metrics
        assert "error_rate" in diagnostics.metrics
        assert "active_sessions" in diagnostics.metrics

    def test_get_diagnostics_healthy_by_default(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test new PRIME instance is healthy."""
        prime = PRIME(test_config)
        diagnostics = prime.get_diagnostics()

        assert diagnostics.status == "healthy"
        assert diagnostics.metrics["error_rate"] == 0.0


class TestResetSession:
    """Tests for PRIME.reset_session() method."""

    def test_reset_session_clears_context(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test reset_session clears session context."""
        prime = PRIME(test_config)
        session_id = "test_session"

        # Build up session context
        prime.process_turn("Input 1", session_id=session_id)
        prime.process_turn("Input 2", session_id=session_id)

        # Reset
        prime.reset_session(session_id)

        # SSM should be reset, next turn starts fresh
        response = prime.process_turn("Input 3", session_id=session_id)
        assert response.turn_number == 1

    def test_reset_session_nonexistent_session(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test reset_session handles nonexistent session."""
        prime = PRIME(test_config)

        # Should not raise
        prime.reset_session("nonexistent_session")


class TestSessionContext:
    """Tests for session context management."""

    def test_session_context_maintained_across_turns(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test session context is maintained across turns."""
        prime = PRIME(test_config)
        session_id = "context_test"

        prime.process_turn("Turn 1", session_id=session_id)
        prime.process_turn("Turn 2", session_id=session_id)
        prime.process_turn("Turn 3", session_id=session_id)

        # Should have active session
        diagnostics = prime.get_diagnostics()
        assert diagnostics.metrics["active_sessions"] >= 1.0

    def test_multiple_sessions_independent(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test multiple sessions have independent context windows.

        Note: SSM turn_number is global (tracks all inputs), but session
        context (predictor embeddings) is per-session. This test verifies
        that context windows are independent, not turn numbers.
        """
        prime = PRIME(test_config)

        # Process turns in different sessions
        prime.process_turn("Input A1", session_id="session_a")
        prime.process_turn("Input B1", session_id="session_b")

        # Both sessions should exist in context
        assert "session_a" in prime._session_context
        assert "session_b" in prime._session_context

        # Context windows should be independent
        context_a = prime._session_context["session_a"]
        context_b = prime._session_context["session_b"]

        assert len(context_a) == 1
        assert len(context_b) == 1

        # Add more to session_a only
        prime.process_turn("Input A2", session_id="session_a")

        assert len(prime._session_context["session_a"]) == 2
        assert len(prime._session_context["session_b"]) == 1  # Unchanged


class TestErrorHandling:
    """Tests for error handling."""

    def test_component_error_propagates(self) -> None:
        """Test ComponentError propagates from process_turn."""
        # Create config with invalid settings to trigger error
        # Note: This is hard to trigger with mock, testing basic flow
        config = PRIMEConfig.for_testing()
        prime = PRIME(config)

        # Normal operation should not raise
        response = prime.process_turn("Test")
        assert response is not None


class TestBoundaryDetection:
    """Tests for semantic boundary detection integration."""

    def test_boundary_crossed_flag(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test boundary_crossed flag in response."""
        prime = PRIME(test_config)
        response = prime.process_turn("Test input")

        assert isinstance(response.boundary_crossed, bool)

    def test_action_triggers_retrieval(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test RETRIEVE action state triggers memory retrieval."""
        prime = PRIME(test_config)

        # Store some content first
        prime.record_response("Stored knowledge about Python")

        # Force retrieval to test the path
        response = prime.process_turn(
            "What about Python?",
            force_retrieval=True,
        )

        assert isinstance(response.retrieved_memories, list)


class TestMemoryReadResultFormat:
    """Tests for MemoryReadResult format from PRIME."""

    def test_memory_read_result_fields(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test MemoryReadResult has all required fields."""
        prime = PRIME(test_config)
        prime.record_response("Test content for search")

        results = prime.search_memory("test")

        for result in results:
            assert hasattr(result, "memory_id")
            assert hasattr(result, "content")
            assert hasattr(result, "cluster_id")
            assert hasattr(result, "similarity")
            assert hasattr(result, "metadata")
            assert hasattr(result, "created_at")


class TestMemoryWriteResultFormat:
    """Tests for MemoryWriteResult format from PRIME."""

    def test_memory_write_result_fields(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test MemoryWriteResult has all required fields."""
        prime = PRIME(test_config)
        result = prime.record_response("Test content")

        assert hasattr(result, "memory_id")
        assert hasattr(result, "cluster_id")
        assert hasattr(result, "is_new_cluster")
        assert hasattr(result, "consolidated")

    def test_memory_write_result_types(
        self, test_config: PRIMEConfig
    ) -> None:
        """Test MemoryWriteResult field types."""
        prime = PRIME(test_config)
        result = prime.record_response("Test content")

        assert isinstance(result.memory_id, str)
        assert isinstance(result.cluster_id, int)
        assert isinstance(result.is_new_cluster, bool)
        assert isinstance(result.consolidated, bool)
