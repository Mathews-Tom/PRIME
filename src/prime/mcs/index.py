"""Vector index protocol and types for MCS.

Defines the VectorIndex protocol for pluggable vector storage backends
and common types for search operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class SparseVector:
    """Sparse vector representation for BM25/keyword search.

    Stores non-zero indices and their corresponding values
    for efficient sparse vector operations.

    Attributes:
        indices: List of non-zero dimension indices.
        values: List of values at the corresponding indices.
    """

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        """Validate indices and values have same length."""
        if len(self.indices) != len(self.values):
            msg = f"indices ({len(self.indices)}) and values ({len(self.values)}) must have same length"
            raise ValueError(msg)


@dataclass
class IndexSearchResult:
    """Result from vector index search.

    Contains the ID and score of a matching vector.

    Attributes:
        id: Unique identifier of the indexed item.
        score: Similarity or relevance score.
    """

    id: str
    score: float


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector index backends.

    Defines the interface for vector storage and search operations
    used by the Memory Cluster Store. Implementations can use
    different backends (Qdrant, FAISS, etc.).

    The protocol supports both dense (semantic) and sparse (keyword)
    vectors for hybrid search capabilities.
    """

    def add(
        self,
        id: str,
        dense: NDArray[np.float32],
        sparse: SparseVector | None = None,
        payload: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Add a vector to the index.

        Args:
            id: Unique identifier for the vector.
            dense: Dense embedding vector (L2-normalized).
            sparse: Optional sparse vector for hybrid search.
            payload: Optional metadata payload.
        """
        ...

    def search_dense(
        self,
        query: NDArray[np.float32],
        top_k: int,
        filter_payload: dict[str, str | int | float | bool] | None = None,
    ) -> list[IndexSearchResult]:
        """Search using dense vector similarity.

        Args:
            query: Query embedding vector.
            top_k: Number of results to return.
            filter_payload: Optional payload filter.

        Returns:
            List of search results ordered by similarity.
        """
        ...

    def search_hybrid(
        self,
        dense_query: NDArray[np.float32],
        sparse_query: SparseVector,
        top_k: int,
        filter_payload: dict[str, str | int | float | bool] | None = None,
    ) -> list[IndexSearchResult]:
        """Search using hybrid dense + sparse with RRF fusion.

        Args:
            dense_query: Dense embedding query vector.
            sparse_query: Sparse vector for keyword matching.
            top_k: Number of results to return.
            filter_payload: Optional payload filter.

        Returns:
            List of search results with fused scores.
        """
        ...

    def remove(self, id: str) -> bool:
        """Remove a vector from the index.

        Args:
            id: Identifier of the vector to remove.

        Returns:
            True if vector was removed, False if not found.
        """
        ...

    def get(self, id: str) -> NDArray[np.float32] | None:
        """Get a vector by ID.

        Args:
            id: Identifier of the vector.

        Returns:
            The dense vector if found, None otherwise.
        """
        ...

    def count(self) -> int:
        """Return the number of vectors in the index.

        Returns:
            Count of indexed vectors.
        """
        ...

    def clear(self) -> None:
        """Remove all vectors from the index."""
        ...
