"""Integration tests for PRIME REST API.

Tests API endpoints with full application stack using httpx.AsyncClient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from prime import PRIMEConfig
from prime.api.app import MiddlewareConfig, create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
def test_config() -> PRIMEConfig:
    """Create PRIME config for integration testing."""
    return PRIMEConfig.for_testing()


@pytest.fixture
def middleware_config_with_auth() -> MiddlewareConfig:
    """Create middleware config with authentication enabled."""
    return MiddlewareConfig(
        enable_cors=True,
        enable_rate_limit=False,
        enable_auth=True,
        api_keys=frozenset({"test-api-key", "another-key"}),
        enable_logging=False,
    )


@pytest.fixture
def middleware_config_with_rate_limit() -> MiddlewareConfig:
    """Create middleware config with rate limiting enabled."""
    return MiddlewareConfig(
        enable_cors=True,
        enable_rate_limit=True,
        requests_per_minute=5,  # Low limit for testing
        rate_limit_window=60,
        enable_auth=False,
        enable_logging=False,
    )


@pytest.fixture
async def client(test_config: PRIMEConfig) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with testing config."""
    app = create_app(config=test_config)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def auth_client(
    test_config: PRIMEConfig,
    middleware_config_with_auth: MiddlewareConfig,
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with auth enabled."""
    app = create_app(config=test_config, middleware_config=middleware_config_with_auth)
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def rate_limit_client(
    test_config: PRIMEConfig,
    middleware_config_with_rate_limit: MiddlewareConfig,
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with rate limiting enabled."""
    app = create_app(
        config=test_config, middleware_config=middleware_config_with_rate_limit
    )
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class TestHealthEndpoint:
    """Tests for health endpoint."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """Test health endpoint returns healthy status."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    async def test_health_no_auth_required(self, auth_client: AsyncClient) -> None:
        """Test health endpoint works without authentication."""
        response = await auth_client.get("/api/v1/health")
        assert response.status_code == 200


class TestProcessEndpoint:
    """Tests for process endpoint."""

    async def test_process_success(self, client: AsyncClient) -> None:
        """Test POST /api/v1/process returns valid response."""
        response = await client.post(
            "/api/v1/process",
            json={"input": "What is machine learning?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "boundary_crossed" in data
        assert "variance" in data
        assert "session_id" in data
        assert "turn_number" in data

    async def test_process_with_session_id(self, client: AsyncClient) -> None:
        """Test process with explicit session ID."""
        session_id = "test-session-123"
        response = await client.post(
            "/api/v1/process",
            json={"input": "Hello world", "session_id": session_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    async def test_process_with_force_retrieval(self, client: AsyncClient) -> None:
        """Test process with force_retrieval flag."""
        response = await client.post(
            "/api/v1/process",
            json={"input": "Test query", "force_retrieval": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "retrieved_memories" in data

    async def test_process_empty_input_rejected(self, client: AsyncClient) -> None:
        """Test empty input is rejected."""
        response = await client.post(
            "/api/v1/process",
            json={"input": ""},
        )
        assert response.status_code == 422

    async def test_process_missing_input_rejected(self, client: AsyncClient) -> None:
        """Test missing input field is rejected."""
        response = await client.post(
            "/api/v1/process",
            json={},
        )
        assert response.status_code == 422


class TestMemoryEndpoints:
    """Tests for memory endpoints."""

    async def test_memory_search(self, client: AsyncClient) -> None:
        """Test POST /api/v1/memory/search."""
        response = await client.post(
            "/api/v1/memory/search",
            json={"query": "test query", "k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    async def test_memory_write(self, client: AsyncClient) -> None:
        """Test POST /api/v1/memory/write."""
        response = await client.post(
            "/api/v1/memory/write",
            json={"content": "Test memory content"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "memory_id" in data
        assert "cluster_id" in data

    async def test_memory_write_with_metadata(self, client: AsyncClient) -> None:
        """Test memory write with metadata."""
        response = await client.post(
            "/api/v1/memory/write",
            json={
                "content": "Memory with metadata",
                "metadata": {"source": "test", "priority": 1},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "memory_id" in data


class TestDiagnosticsEndpoint:
    """Tests for diagnostics endpoint."""

    async def test_diagnostics_returns_info(self, client: AsyncClient) -> None:
        """Test GET /api/v1/diagnostics returns system info."""
        response = await client.get("/api/v1/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data


class TestClustersEndpoint:
    """Tests for clusters endpoint."""

    async def test_clusters_list(self, client: AsyncClient) -> None:
        """Test GET /api/v1/clusters."""
        response = await client.get("/api/v1/clusters")
        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert isinstance(data["clusters"], list)


class TestAuthentication:
    """Tests for API authentication."""

    async def test_unauthorized_without_key(self, auth_client: AsyncClient) -> None:
        """Test request without API key is rejected."""
        response = await auth_client.post(
            "/api/v1/process",
            json={"input": "test"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "error_code" in data

    async def test_authorized_with_valid_key(self, auth_client: AsyncClient) -> None:
        """Test request with valid API key succeeds."""
        response = await auth_client.post(
            "/api/v1/process",
            json={"input": "test"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200

    async def test_unauthorized_with_invalid_key(
        self, auth_client: AsyncClient
    ) -> None:
        """Test request with invalid API key is rejected."""
        response = await auth_client.post(
            "/api/v1/process",
            json={"input": "test"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401


class TestRateLimiting:
    """Tests for rate limiting."""

    async def test_rate_limit_headers_present(
        self, rate_limit_client: AsyncClient
    ) -> None:
        """Test rate limit headers are included in response."""
        response = await rate_limit_client.post(
            "/api/v1/process",
            json={"input": "test"},
        )
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    async def test_rate_limit_decrements(self, rate_limit_client: AsyncClient) -> None:
        """Test rate limit remaining decrements with each request."""
        response1 = await rate_limit_client.post(
            "/api/v1/process",
            json={"input": "test 1"},
        )
        remaining1 = int(response1.headers["X-RateLimit-Remaining"])

        response2 = await rate_limit_client.post(
            "/api/v1/process",
            json={"input": "test 2"},
        )
        remaining2 = int(response2.headers["X-RateLimit-Remaining"])

        assert remaining2 < remaining1

    async def test_rate_limit_enforced(self, rate_limit_client: AsyncClient) -> None:
        """Test rate limiting blocks requests after threshold."""
        # Make requests up to limit (5 requests)
        for i in range(5):
            response = await rate_limit_client.post(
                "/api/v1/process",
                json={"input": f"test {i}"},
            )
            assert response.status_code == 200

        # Next request should be rate limited
        response = await rate_limit_client.post(
            "/api/v1/process",
            json={"input": "over limit"},
        )
        assert response.status_code == 429
        assert "Retry-After" in response.headers


class TestEndToEnd:
    """End-to-end workflow tests."""

    async def test_write_then_search_memory(self, client: AsyncClient) -> None:
        """Test writing memory then searching for it."""
        # Write memory
        write_response = await client.post(
            "/api/v1/memory/write",
            json={"content": "Python is a programming language"},
        )
        assert write_response.status_code == 200

        # Search for it
        search_response = await client.post(
            "/api/v1/memory/search",
            json={"query": "programming languages", "k": 5},
        )
        assert search_response.status_code == 200

    async def test_session_continuity(self, client: AsyncClient) -> None:
        """Test conversation session maintains state."""
        session_id = "continuity-test-session"

        # First turn
        response1 = await client.post(
            "/api/v1/process",
            json={"input": "Hello, I want to learn about AI", "session_id": session_id},
        )
        assert response1.status_code == 200
        assert response1.json()["turn_number"] == 1

        # Second turn
        response2 = await client.post(
            "/api/v1/process",
            json={"input": "Tell me more about neural networks", "session_id": session_id},
        )
        assert response2.status_code == 200
        assert response2.json()["turn_number"] == 2

        # Third turn
        response3 = await client.post(
            "/api/v1/process",
            json={"input": "What about deep learning?", "session_id": session_id},
        )
        assert response3.status_code == 200
        assert response3.json()["turn_number"] == 3
