"""Unit tests for PRIME client SDK.

Tests PRIMEClient and PRIMEClientSync with mocked HTTP responses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from prime.client import (
    HealthStatus,
    MemoryResult,
    PRIMEClient,
    PRIMEClientSync,
    ProcessResponse,
    SearchResponse,
    WriteResult,
)
from prime.exceptions import AuthenticationError, PRIMEError, RateLimitError

if TYPE_CHECKING:
    pass


def create_mock_transport(responses: dict[str, httpx.Response]) -> httpx.MockTransport:
    """Create mock transport with predefined responses.

    Args:
        responses: Dict mapping request paths to responses.

    Returns:
        MockTransport for testing.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in responses:
            return responses[path]
        return httpx.Response(404, json={"detail": "Not found"})

    return httpx.MockTransport(handler)


class TestPRIMEClientCreation:
    """Tests for PRIMEClient initialization."""

    def test_client_creation_default(self) -> None:
        """Test client creation with defaults."""
        client = PRIMEClient()
        assert client.base_url == "http://localhost:8000"

    def test_client_creation_custom_url(self) -> None:
        """Test client creation with custom URL."""
        client = PRIMEClient(base_url="http://api.example.com:9000")
        assert client.base_url == "http://api.example.com:9000"

    def test_client_creation_strips_trailing_slash(self) -> None:
        """Test client strips trailing slash from URL."""
        client = PRIMEClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_client_creation_with_api_key(self) -> None:
        """Test client creation with API key."""
        client = PRIMEClient(api_key="test-api-key")
        assert client._headers["X-API-Key"] == "test-api-key"


class TestPRIMEClientAsync:
    """Tests for async PRIMEClient methods."""

    @pytest.mark.asyncio
    async def test_process_turn_success(self) -> None:
        """Test successful process_turn call."""
        response_data = {
            "retrieved_memories": [
                {
                    "memory_id": "mem-1",
                    "content": "Test content",
                    "cluster_id": 1,
                    "similarity": 0.95,
                    "metadata": {},
                    "created_at": 1000.0,
                }
            ],
            "boundary_crossed": True,
            "variance": 0.25,
            "smoothed_variance": 0.20,
            "action": "retrieve",
            "session_id": "test-session",
            "turn_number": 5,
            "latency_ms": 15.5,
        }

        transport = create_mock_transport(
            {"/api/v1/process": httpx.Response(200, json=response_data)}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            result = await client.process_turn("What is Python?")

            assert isinstance(result, ProcessResponse)
            assert result.action == "retrieve"
            assert result.boundary_crossed is True
            assert len(result.retrieved_memories) == 1
            assert result.retrieved_memories[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_process_turn_with_options(self) -> None:
        """Test process_turn with all options."""
        response_data = {
            "retrieved_memories": [],
            "boundary_crossed": False,
            "variance": 0.05,
            "smoothed_variance": 0.04,
            "action": "continue",
            "session_id": "custom-session",
            "turn_number": 1,
            "latency_ms": 5.0,
        }

        requests_made: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_made.append(request)
            return httpx.Response(200, json=response_data)

        transport = httpx.MockTransport(handler)

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            await client.process_turn(
                "Test query",
                session_id="custom-session",
                user_id="user-123",
                force_retrieval=True,
                k=10,
            )

            assert len(requests_made) == 1
            body = json.loads(requests_made[0].content)
            assert body["input"] == "Test query"
            assert body["session_id"] == "custom-session"
            assert body["user_id"] == "user-123"
            assert body["force_retrieval"] is True
            assert body["k"] == 10

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        """Test successful search call."""
        response_data = {
            "results": [
                {
                    "memory_id": "mem-1",
                    "content": "Search result",
                    "cluster_id": 2,
                    "similarity": 0.88,
                    "metadata": {"source": "docs"},
                    "created_at": 999.0,
                }
            ],
            "query_embedding": None,
        }

        transport = create_mock_transport(
            {"/api/v1/memory/search": httpx.Response(200, json=response_data)}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            result = await client.search("test query")

            assert isinstance(result, SearchResponse)
            assert len(result.results) == 1
            assert result.results[0].memory_id == "mem-1"
            assert result.results[0].similarity == 0.88

    @pytest.mark.asyncio
    async def test_write_memory_success(self) -> None:
        """Test successful write_memory call."""
        response_data = {
            "memory_id": "new-mem-123",
            "cluster_id": 5,
            "is_new_cluster": True,
            "consolidated": False,
        }

        transport = create_mock_transport(
            {"/api/v1/memory/write": httpx.Response(200, json=response_data)}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            result = await client.write_memory(
                "New memory content",
                metadata={"source": "test"},
            )

            assert isinstance(result, WriteResult)
            assert result.memory_id == "new-mem-123"
            assert result.cluster_id == 5
            assert result.is_new_cluster is True

    @pytest.mark.asyncio
    async def test_health_success(self) -> None:
        """Test successful health call."""
        response_data = {"status": "healthy", "version": "1.0.0"}

        transport = create_mock_transport(
            {"/api/v1/health": httpx.Response(200, json=response_data)}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            result = await client.health()

            assert isinstance(result, HealthStatus)
            assert result.status == "healthy"
            assert result.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_diagnostics_success(self) -> None:
        """Test successful diagnostics call."""
        response_data = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
            "components": {},
            "metrics": {},
        }

        transport = create_mock_transport(
            {"/api/v1/diagnostics": httpx.Response(200, json=response_data)}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            result = await client.diagnostics()

            assert isinstance(result, dict)
            assert result["status"] == "healthy"


class TestPRIMEClientErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_authentication_error_401(self) -> None:
        """Test 401 response raises AuthenticationError."""
        transport = create_mock_transport(
            {"/api/v1/health": httpx.Response(401, json={"detail": "Invalid API key"})}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            with pytest.raises(AuthenticationError) as exc_info:
                await client.health()

            assert "Invalid or missing API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authentication_error_403(self) -> None:
        """Test 403 response raises AuthenticationError."""
        transport = create_mock_transport(
            {"/api/v1/health": httpx.Response(403, json={"detail": "Access denied"})}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            with pytest.raises(AuthenticationError) as exc_info:
                await client.health()

            assert "Access denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        """Test 429 response raises RateLimitError."""
        transport = create_mock_transport(
            {
                "/api/v1/health": httpx.Response(
                    429,
                    json={"detail": "Too many requests"},
                    headers={"Retry-After": "30"},
                )
            }
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            with pytest.raises(RateLimitError) as exc_info:
                await client.health()

            assert exc_info.value.retry_after == 30

    @pytest.mark.asyncio
    async def test_generic_error(self) -> None:
        """Test other error responses raise PRIMEError."""
        transport = create_mock_transport(
            {"/api/v1/health": httpx.Response(500, json={"detail": "Server error"})}
        )

        async with httpx.AsyncClient(
            base_url="http://test", transport=transport
        ) as http_client:
            client = PRIMEClient()
            client._client = http_client

            with pytest.raises(PRIMEError) as exc_info:
                await client.health()

            assert "500" in str(exc_info.value)
            assert exc_info.value.error_code == "PRIME_HTTP_500"


class TestPRIMEClientContextManager:
    """Tests for context manager protocol."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager works."""
        async with PRIMEClient() as client:
            assert isinstance(client, PRIMEClient)


class TestPRIMEClientSyncCreation:
    """Tests for PRIMEClientSync initialization."""

    def test_sync_client_creation(self) -> None:
        """Test sync client creation."""
        client = PRIMEClientSync()
        assert client._async_client.base_url == "http://localhost:8000"

    def test_sync_client_with_options(self) -> None:
        """Test sync client creation with options."""
        client = PRIMEClientSync(
            base_url="http://api.example.com",
            api_key="test-key",
            timeout=60.0,
        )
        assert client._async_client.base_url == "http://api.example.com"
        assert client._async_client._headers["X-API-Key"] == "test-key"


class TestDataClasses:
    """Tests for response data classes."""

    def test_memory_result_creation(self) -> None:
        """Test MemoryResult dataclass."""
        result = MemoryResult(
            memory_id="mem-1",
            content="Test content",
            cluster_id=1,
            similarity=0.95,
            metadata={"key": "value"},
            created_at=1000.0,
        )
        assert result.memory_id == "mem-1"
        assert result.similarity == 0.95

    def test_process_response_from_dict(self) -> None:
        """Test ProcessResponse.from_dict factory."""
        data = {
            "retrieved_memories": [
                {
                    "memory_id": "mem-1",
                    "content": "Content",
                    "cluster_id": 1,
                    "similarity": 0.9,
                    "metadata": {},
                    "created_at": 0.0,
                }
            ],
            "boundary_crossed": True,
            "variance": 0.2,
            "smoothed_variance": 0.18,
            "action": "retrieve",
            "session_id": "sess-1",
            "turn_number": 3,
            "latency_ms": 10.0,
        }

        response = ProcessResponse.from_dict(data)
        assert response.action == "retrieve"
        assert len(response.retrieved_memories) == 1
        assert response.retrieved_memories[0].memory_id == "mem-1"

    def test_search_response_from_dict(self) -> None:
        """Test SearchResponse.from_dict factory."""
        data = {
            "results": [
                {
                    "memory_id": "mem-1",
                    "content": "Result",
                    "cluster_id": 2,
                    "similarity": 0.85,
                    "metadata": {},
                    "created_at": 0.0,
                }
            ],
            "query_embedding": [0.1, 0.2, 0.3],
        }

        response = SearchResponse.from_dict(data)
        assert len(response.results) == 1
        assert response.query_embedding == [0.1, 0.2, 0.3]

    def test_write_result_creation(self) -> None:
        """Test WriteResult dataclass."""
        result = WriteResult(
            memory_id="mem-1",
            cluster_id=5,
            is_new_cluster=True,
            consolidated=False,
        )
        assert result.memory_id == "mem-1"
        assert result.is_new_cluster is True

    def test_health_status_creation(self) -> None:
        """Test HealthStatus dataclass."""
        status = HealthStatus(status="healthy", version="1.0.0")
        assert status.status == "healthy"
        assert status.version == "1.0.0"


class TestClientExports:
    """Tests for client module exports."""

    def test_exports_available(self) -> None:
        """Test all exports are available."""
        from prime import client

        assert hasattr(client, "PRIMEClient")
        assert hasattr(client, "PRIMEClientSync")
        assert hasattr(client, "MemoryResult")
        assert hasattr(client, "ProcessResponse")
        assert hasattr(client, "SearchResponse")
        assert hasattr(client, "WriteResult")
        assert hasattr(client, "HealthStatus")
