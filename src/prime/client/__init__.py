"""PRIME Python SDK Client.

Provides async and sync clients for interacting with the PRIME REST API.

Example (async):
    >>> from prime.client import PRIMEClient
    >>> async with PRIMEClient(api_key="sk-...") as client:
    ...     result = await client.process_turn("What is JEPA?")

Example (sync):
    >>> from prime.client import PRIMEClientSync
    >>> with PRIMEClientSync(api_key="sk-...") as client:
    ...     result = client.process_turn("What is JEPA?")
"""

from __future__ import annotations

from prime.client.client import (
    HealthStatus,
    MemoryResult,
    PRIMEClient,
    PRIMEClientSync,
    ProcessResponse,
    SearchResponse,
    WriteResult,
)

__all__ = [
    "HealthStatus",
    "MemoryResult",
    "PRIMEClient",
    "PRIMEClientSync",
    "ProcessResponse",
    "SearchResponse",
    "WriteResult",
]
