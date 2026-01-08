"""PRIME Python SDK Client.

Provides async and sync clients for interacting with the PRIME REST API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx

from prime.exceptions import AuthenticationError, PRIMEError, RateLimitError

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from types import TracebackType

_T = TypeVar("_T")


@dataclass(frozen=True)
class MemoryResult:
    """Memory result from API.

    Attributes:
        memory_id: Unique memory identifier.
        content: Memory content.
        cluster_id: Cluster the memory belongs to.
        similarity: Similarity score (0-1).
        metadata: Additional metadata.
        created_at: Creation timestamp.
    """

    memory_id: str
    content: str
    cluster_id: int
    similarity: float
    metadata: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class ProcessResponse:
    """Response from process_turn.

    Attributes:
        retrieved_memories: Retrieved memory results.
        boundary_crossed: Whether semantic boundary was crossed.
        variance: Current variance value.
        smoothed_variance: Smoothed variance value.
        action: Action taken (continue/prepare/retrieve).
        session_id: Session identifier.
        turn_number: Current turn number.
        latency_ms: Processing latency in milliseconds.
    """

    retrieved_memories: list[MemoryResult]
    boundary_crossed: bool
    variance: float
    smoothed_variance: float
    action: str
    session_id: str
    turn_number: int
    latency_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessResponse:
        """Create from API response dict."""
        return cls(
            retrieved_memories=[
                MemoryResult(**m) for m in data.get("retrieved_memories", [])
            ],
            boundary_crossed=data.get("boundary_crossed", False),
            variance=data.get("variance", 0.0),
            smoothed_variance=data.get("smoothed_variance", 0.0),
            action=data.get("action", "continue"),
            session_id=data.get("session_id", ""),
            turn_number=data.get("turn_number", 0),
            latency_ms=data.get("latency_ms", 0.0),
        )


@dataclass(frozen=True)
class WriteResult:
    """Result from write_memory.

    Attributes:
        memory_id: ID of the written memory.
        cluster_id: Cluster the memory was assigned to.
        is_new_cluster: Whether a new cluster was created.
        consolidated: Whether consolidation occurred.
    """

    memory_id: str
    cluster_id: int
    is_new_cluster: bool
    consolidated: bool


@dataclass(frozen=True)
class SearchResponse:
    """Response from memory search.

    Attributes:
        results: List of memory results.
        query_embedding: Optional query embedding vector.
    """

    results: list[MemoryResult]
    query_embedding: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResponse:
        """Create from API response dict."""
        return cls(
            results=[MemoryResult(**m) for m in data.get("results", [])],
            query_embedding=data.get("query_embedding"),
        )


@dataclass(frozen=True)
class HealthStatus:
    """API health status.

    Attributes:
        status: Health status string.
        version: API version.
    """

    status: str
    version: str


class PRIMEClient:
    """Async Python SDK for PRIME API.

    Provides async methods for interacting with the PRIME REST API.
    Supports context manager protocol for automatic connection cleanup.

    Attributes:
        base_url: API base URL.

    Example:
        >>> async with PRIMEClient(api_key="sk-...") as client:
        ...     result = await client.process_turn("What is JEPA?")
        ...     print(f"Action: {result.action}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: API base URL (default: http://localhost:8000).
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds (default: 30.0).
        """
        self.base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=timeout,
        )

    async def __aenter__(self) -> PRIMEClient:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self._client.aclose()

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses.

        Args:
            response: HTTP response.

        Raises:
            AuthenticationError: For 401/403 responses.
            RateLimitError: For 429 responses.
            PRIMEError: For other error responses.
        """
        if response.status_code == 401:
            raise AuthenticationError("Invalid or missing API key")
        if response.status_code == 403:
            raise AuthenticationError("Access denied")
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(retry_after)
        if response.status_code >= 400:
            try:
                error_data = response.json()
                detail = error_data.get("detail", response.text)
            except Exception:
                detail = response.text
            raise PRIMEError(
                f"API error ({response.status_code}): {detail}",
                f"PRIME_HTTP_{response.status_code}",
            )

    async def process_turn(
        self,
        content: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        force_retrieval: bool = False,
        k: int = 5,
    ) -> ProcessResponse:
        """Process a conversation turn.

        Args:
            content: Input content/query.
            session_id: Optional session ID for stateful conversations.
            user_id: Optional user identifier.
            force_retrieval: Force retrieval regardless of boundary detection.
            k: Number of memories to retrieve (default: 5).

        Returns:
            ProcessResponse with retrieval results and metadata.

        Raises:
            PRIMEError: If the API request fails.
        """
        payload: dict[str, Any] = {
            "input": content,
            "force_retrieval": force_retrieval,
            "k": k,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if user_id is not None:
            payload["user_id"] = user_id

        response = await self._client.post("/api/v1/process", json=payload)
        if response.status_code >= 400:
            self._handle_error(response)
        return ProcessResponse.from_dict(response.json())

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
        min_similarity: float = 0.0,
    ) -> SearchResponse:
        """Search memory directly.

        Args:
            query: Search query.
            k: Number of results to return (default: 5).
            user_id: Optional user filter.
            session_id: Optional session filter.
            min_similarity: Minimum similarity threshold (default: 0.0).

        Returns:
            SearchResponse with memory results.

        Raises:
            PRIMEError: If the API request fails.
        """
        payload: dict[str, Any] = {
            "query": query,
            "k": k,
            "min_similarity": min_similarity,
        }
        if user_id is not None:
            payload["user_id"] = user_id
        if session_id is not None:
            payload["session_id"] = session_id

        response = await self._client.post("/api/v1/memory/search", json=payload)
        if response.status_code >= 400:
            self._handle_error(response)
        return SearchResponse.from_dict(response.json())

    async def write_memory(
        self,
        content: str,
        *,
        metadata: dict[str, str | int | float | bool] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> WriteResult:
        """Write content to memory.

        Args:
            content: Content to store.
            metadata: Optional metadata dictionary.
            user_id: Optional user identifier.
            session_id: Optional session identifier.

        Returns:
            WriteResult with memory ID and cluster info.

        Raises:
            PRIMEError: If the API request fails.
        """
        payload: dict[str, Any] = {"content": content}
        if metadata is not None:
            payload["metadata"] = metadata
        if user_id is not None:
            payload["user_id"] = user_id
        if session_id is not None:
            payload["session_id"] = session_id

        response = await self._client.post("/api/v1/memory/write", json=payload)
        if response.status_code >= 400:
            self._handle_error(response)
        data = response.json()
        return WriteResult(
            memory_id=data["memory_id"],
            cluster_id=data["cluster_id"],
            is_new_cluster=data["is_new_cluster"],
            consolidated=data["consolidated"],
        )

    async def health(self) -> HealthStatus:
        """Check API health.

        Returns:
            HealthStatus with status and version.

        Raises:
            PRIMEError: If the API request fails.
        """
        response = await self._client.get("/api/v1/health")
        if response.status_code >= 400:
            self._handle_error(response)
        data = response.json()
        return HealthStatus(status=data["status"], version=data["version"])

    async def diagnostics(self) -> dict[str, Any]:
        """Get system diagnostics.

        Returns:
            Diagnostics dictionary with system information.

        Raises:
            PRIMEError: If the API request fails.
        """
        response = await self._client.get("/api/v1/diagnostics")
        if response.status_code >= 400:
            self._handle_error(response)
        return cast("dict[str, Any]", response.json())


class PRIMEClientSync:
    """Synchronous wrapper for PRIMEClient.

    Provides synchronous methods by wrapping the async client.
    Supports context manager protocol for automatic connection cleanup.

    Example:
        >>> with PRIMEClientSync(api_key="sk-...") as client:
        ...     result = client.process_turn("What is JEPA?")
        ...     print(f"Action: {result.action}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the synchronous client.

        Args:
            base_url: API base URL (default: http://localhost:8000).
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds (default: 30.0).
        """
        self._async_client = PRIMEClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def __enter__(self) -> PRIMEClientSync:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client connection."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_client.close())
            loop.close()
        else:
            loop.run_until_complete(self._async_client.close())

    def _run_async(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run an async coroutine synchronously.

        Args:
            coro: Coroutine to run.

        Returns:
            Coroutine result.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        else:
            return loop.run_until_complete(coro)

    def process_turn(
        self,
        content: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        force_retrieval: bool = False,
        k: int = 5,
    ) -> ProcessResponse:
        """Process a conversation turn synchronously.

        Args:
            content: Input content/query.
            session_id: Optional session ID for stateful conversations.
            user_id: Optional user identifier.
            force_retrieval: Force retrieval regardless of boundary detection.
            k: Number of memories to retrieve (default: 5).

        Returns:
            ProcessResponse with retrieval results and metadata.
        """
        return self._run_async(
            self._async_client.process_turn(
                content,
                session_id=session_id,
                user_id=user_id,
                force_retrieval=force_retrieval,
                k=k,
            )
        )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
        min_similarity: float = 0.0,
    ) -> SearchResponse:
        """Search memory directly synchronously.

        Args:
            query: Search query.
            k: Number of results to return (default: 5).
            user_id: Optional user filter.
            session_id: Optional session filter.
            min_similarity: Minimum similarity threshold (default: 0.0).

        Returns:
            SearchResponse with memory results.
        """
        return self._run_async(
            self._async_client.search(
                query,
                k=k,
                user_id=user_id,
                session_id=session_id,
                min_similarity=min_similarity,
            )
        )

    def write_memory(
        self,
        content: str,
        *,
        metadata: dict[str, str | int | float | bool] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> WriteResult:
        """Write content to memory synchronously.

        Args:
            content: Content to store.
            metadata: Optional metadata dictionary.
            user_id: Optional user identifier.
            session_id: Optional session identifier.

        Returns:
            WriteResult with memory ID and cluster info.
        """
        return self._run_async(
            self._async_client.write_memory(
                content,
                metadata=metadata,
                user_id=user_id,
                session_id=session_id,
            )
        )

    def health(self) -> HealthStatus:
        """Check API health synchronously.

        Returns:
            HealthStatus with status and version.
        """
        return self._run_async(self._async_client.health())

    def diagnostics(self) -> dict[str, Any]:
        """Get system diagnostics synchronously.

        Returns:
            Diagnostics dictionary with system information.
        """
        return self._run_async(self._async_client.diagnostics())
