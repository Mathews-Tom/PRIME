"""Unit tests for PRIME API endpoints.

Tests all FastAPI endpoints using TestClient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from prime import PRIMEConfig
from prime.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create test client with testing configuration.

    Uses context manager to ensure lifespan events (startup/shutdown) are triggered.
    """
    config = PRIMEConfig.for_testing()
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for GET /api/v1/health endpoint."""

    def test_health_check_returns_healthy(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestDiagnosticsEndpoint:
    """Tests for GET /api/v1/diagnostics endpoint."""

    def test_diagnostics_returns_system_info(self, client: TestClient) -> None:
        """Test diagnostics endpoint returns system information."""
        response = client.get("/api/v1/diagnostics")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["uptime_seconds"] >= 0
        assert "components" in data
        assert "metrics" in data

    def test_diagnostics_includes_components(self, client: TestClient) -> None:
        """Test diagnostics includes all component status."""
        response = client.get("/api/v1/diagnostics")
        data = response.json()

        components = data["components"]
        assert "ssm" in components
        assert "mcs" in components
        assert "predictor" in components
        assert "y_encoder" in components

        # Each component should have required fields
        for _, status in components.items():
            assert "name" in status
            assert "status" in status
            assert "latency_p50_ms" in status
            assert "error_rate" in status


class TestProcessEndpoint:
    """Tests for POST /api/v1/process endpoint."""

    def test_process_turn_basic(self, client: TestClient) -> None:
        """Test basic conversation turn processing."""
        response = client.post(
            "/api/v1/process",
            json={"input": "What is machine learning?"},
        )
        assert response.status_code == 200
        data = response.json()

        assert "retrieved_memories" in data
        assert "boundary_crossed" in data
        assert "variance" in data
        assert "smoothed_variance" in data
        assert "action" in data
        assert "session_id" in data
        assert "turn_number" in data
        assert "latency_ms" in data

    def test_process_turn_with_session_id(self, client: TestClient) -> None:
        """Test conversation turn with custom session ID."""
        response = client.post(
            "/api/v1/process",
            json={
                "input": "Tell me about neural networks",
                "session_id": "test-session-123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_process_turn_generates_session_id(self, client: TestClient) -> None:
        """Test that session ID is generated if not provided."""
        response = client.post(
            "/api/v1/process",
            json={"input": "Hello world"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    def test_process_turn_increments_turn_number(self, client: TestClient) -> None:
        """Test that turn number increments across requests."""
        # First turn
        response1 = client.post(
            "/api/v1/process",
            json={"input": "First message", "session_id": "counting-session"},
        )
        turn1 = response1.json()["turn_number"]

        # Second turn
        response2 = client.post(
            "/api/v1/process",
            json={"input": "Second message", "session_id": "counting-session"},
        )
        turn2 = response2.json()["turn_number"]

        assert turn2 > turn1

    def test_process_turn_validates_empty_input(self, client: TestClient) -> None:
        """Test that empty input is rejected."""
        response = client.post(
            "/api/v1/process",
            json={"input": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_process_turn_with_custom_k(self, client: TestClient) -> None:
        """Test processing with custom k parameter."""
        response = client.post(
            "/api/v1/process",
            json={"input": "Test query", "k": 10},
        )
        assert response.status_code == 200


class TestMemoryWriteEndpoint:
    """Tests for POST /api/v1/memory/write endpoint."""

    def test_write_memory_basic(self, client: TestClient) -> None:
        """Test basic memory write operation."""
        response = client.post(
            "/api/v1/memory/write",
            json={"content": "Machine learning is a subset of artificial intelligence."},
        )
        assert response.status_code == 200
        data = response.json()

        assert "memory_id" in data
        assert "cluster_id" in data
        assert "is_new_cluster" in data
        assert "consolidated" in data

    def test_write_memory_with_metadata(self, client: TestClient) -> None:
        """Test memory write with metadata."""
        response = client.post(
            "/api/v1/memory/write",
            json={
                "content": "Deep learning uses neural networks with many layers.",
                "metadata": {"source": "textbook", "chapter": 1},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["memory_id"] is not None

    def test_write_memory_validates_empty_content(self, client: TestClient) -> None:
        """Test that empty content is rejected."""
        response = client.post(
            "/api/v1/memory/write",
            json={"content": ""},
        )
        assert response.status_code == 422  # Validation error


class TestMemorySearchEndpoint:
    """Tests for POST /api/v1/memory/search endpoint."""

    def test_search_memory_basic(self, client: TestClient) -> None:
        """Test basic memory search."""
        # First write some content
        client.post(
            "/api/v1/memory/write",
            json={"content": "Python is a programming language."},
        )

        # Then search
        response = client.post(
            "/api/v1/memory/search",
            json={"query": "programming languages"},
        )
        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_memory_with_k(self, client: TestClient) -> None:
        """Test search with custom k parameter."""
        response = client.post(
            "/api/v1/memory/search",
            json={"query": "test query", "k": 3},
        )
        assert response.status_code == 200

    def test_search_memory_validates_empty_query(self, client: TestClient) -> None:
        """Test that empty query is rejected."""
        response = client.post(
            "/api/v1/memory/search",
            json={"query": ""},
        )
        assert response.status_code == 422  # Validation error


class TestClustersEndpoint:
    """Tests for /api/v1/clusters endpoints."""

    def test_list_clusters_empty(self, client: TestClient) -> None:
        """Test listing clusters when none exist."""
        response = client.get("/api/v1/clusters")
        assert response.status_code == 200
        data = response.json()

        assert "clusters" in data
        assert "total_count" in data
        assert isinstance(data["clusters"], list)

    def test_list_clusters_after_write(self, client: TestClient) -> None:
        """Test listing clusters after writing memory."""
        # Write some memory to create a cluster
        client.post(
            "/api/v1/memory/write",
            json={"content": "Quantum computing uses qubits."},
        )

        response = client.get("/api/v1/clusters")
        assert response.status_code == 200
        data = response.json()

        assert len(data["clusters"]) > 0

        # Check cluster info structure
        cluster = data["clusters"][0]
        assert "cluster_id" in cluster
        assert "size" in cluster
        assert "is_consolidated" in cluster
        assert "prototype_norm" in cluster

    def test_get_cluster_not_found(self, client: TestClient) -> None:
        """Test getting non-existent cluster."""
        response = client.get("/api/v1/clusters/99999")
        assert response.status_code == 404

    def test_get_cluster_by_id(self, client: TestClient) -> None:
        """Test getting cluster by ID."""
        # Write memory to create a cluster
        write_response = client.post(
            "/api/v1/memory/write",
            json={"content": "Blockchain is a distributed ledger technology."},
        )
        cluster_id = write_response.json()["cluster_id"]

        # Get the cluster
        response = client.get(f"/api/v1/clusters/{cluster_id}")
        assert response.status_code == 200
        data = response.json()

        assert data["cluster_id"] == cluster_id
        assert data["size"] >= 1


class TestConfigEndpoint:
    """Tests for PUT /api/v1/config endpoint."""

    def test_update_config_not_supported(self, client: TestClient) -> None:
        """Test that config update returns not_supported status."""
        response = client.put(
            "/api/v1/config",
            json={"variance_threshold": 0.2},
        )
        assert response.status_code == 200
        data = response.json()

        assert "updated" in data
        assert "changes" in data
        # Config update returns tracking info but doesn't apply yet
        assert "variance_threshold" in data["changes"]
        assert data["changes"]["variance_threshold"]["status"] == "not_supported"


class TestErrorHandling:
    """Tests for error handling in API endpoints."""

    def test_invalid_json_body(self, client: TestClient) -> None:
        """Test handling of invalid JSON body."""
        response = client.post(
            "/api/v1/process",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self, client: TestClient) -> None:
        """Test handling of missing required field."""
        response = client.post(
            "/api/v1/process",
            json={},  # Missing 'input' field
        )
        assert response.status_code == 422


class TestAPIIntegration:
    """Integration tests for API workflow."""

    def test_full_conversation_flow(self, client: TestClient) -> None:
        """Test complete conversation flow through API."""
        session_id = "integration-test-session"

        # Turn 1: Initial query
        response1 = client.post(
            "/api/v1/process",
            json={
                "input": "Tell me about data structures",
                "session_id": session_id,
            },
        )
        assert response1.status_code == 200

        # Turn 2: Follow-up
        response2 = client.post(
            "/api/v1/process",
            json={
                "input": "What about linked lists specifically?",
                "session_id": session_id,
            },
        )
        assert response2.status_code == 200

        # Verify session context is maintained
        data1 = response1.json()
        data2 = response2.json()
        assert data1["session_id"] == data2["session_id"]
        assert data2["turn_number"] > data1["turn_number"]

    def test_write_and_retrieve_memory(self, client: TestClient) -> None:
        """Test writing memory and then searching for it."""
        # Write unique content
        content = "The Fibonacci sequence is 1, 1, 2, 3, 5, 8, 13"
        client.post(
            "/api/v1/memory/write",
            json={"content": content},
        )

        # Search for it
        response = client.post(
            "/api/v1/memory/search",
            json={"query": "Fibonacci sequence numbers"},
        )
        assert response.status_code == 200
        results = response.json()["results"]

        # Should find the memory we just wrote
        # (exact match depends on embedding model)
        assert isinstance(results, list)
