"""In-memory vector index for testing and development.

Provides a simple VectorIndex implementation that stores vectors
in memory without requiring external services like Qdrant.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prime.mcs.index import IndexSearchResult, SparseVector


class MemoryIndex:
    """In-memory vector index for testing and development.

    A simple VectorIndex implementation that stores vectors in memory
    and performs brute-force cosine similarity search. Suitable for
    testing and small-scale development use.

    Not recommended for production with large vector collections.

    Example:
        >>> index = MemoryIndex(embedding_dim=384)
        >>> index.add(id="mem_1", dense=embedding)
        >>> results = index.search_dense(query=query_emb, top_k=5)
    """

    __slots__ = (
        "_embedding_dim",
        "_payloads",
        "_sparse",
        "_vectors",
    )

    def __init__(self, embedding_dim: int = 384) -> None:
        """Initialize in-memory index.

        Args:
            embedding_dim: Dimension of vectors to store.
        """
        self._embedding_dim = embedding_dim
        self._vectors: dict[str, np.ndarray] = {}
        self._sparse: dict[str, SparseVector] = {}
        self._payloads: dict[str, dict[str, Any]] = {}

    def add(
        self,
        id: str,
        dense: np.ndarray,
        sparse: SparseVector | None = None,
        payload: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Add a vector to the index.

        Args:
            id: Unique identifier for the vector.
            dense: Dense embedding vector.
            sparse: Optional sparse vector for hybrid search.
            payload: Optional metadata payload.
        """
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
        """Search using dense vector similarity.

        Args:
            query: Query embedding vector.
            top_k: Number of results to return.
            filter_payload: Optional payload filter.

        Returns:
            List of search results ordered by similarity.
        """
        results: list[tuple[str, float]] = []

        for vec_id, vec in self._vectors.items():
            # Apply filter
            if filter_payload:
                payload = self._payloads.get(vec_id, {})
                if not all(payload.get(k) == v for k, v in filter_payload.items()):
                    continue

            # Cosine similarity (assuming normalized vectors)
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
        sparse_query: SparseVector,
        top_k: int,
        filter_payload: dict[str, str | int | float | bool] | None = None,
    ) -> list[IndexSearchResult]:
        """Search using hybrid dense + sparse (falls back to dense only).

        In-memory index does not support true sparse search.
        This implementation uses dense search only.

        Args:
            dense_query: Dense embedding query vector.
            sparse_query: Sparse vector (ignored in this implementation).
            top_k: Number of results to return.
            filter_payload: Optional payload filter.

        Returns:
            List of search results from dense search.
        """
        del sparse_query  # Not supported in memory index
        return self.search_dense(
            query=dense_query, top_k=top_k, filter_payload=filter_payload
        )

    def remove(self, id: str) -> bool:
        """Remove a vector from the index.

        Args:
            id: Identifier of the vector to remove.

        Returns:
            True if vector was removed, False if not found.
        """
        if id in self._vectors:
            del self._vectors[id]
            self._sparse.pop(id, None)
            self._payloads.pop(id, None)
            return True
        return False

    def get(self, id: str) -> np.ndarray | None:
        """Get a vector by ID.

        Args:
            id: Identifier of the vector.

        Returns:
            The dense vector if found, None otherwise.
        """
        return self._vectors.get(id)

    def count(self) -> int:
        """Return the number of vectors in the index.

        Returns:
            Count of indexed vectors.
        """
        return len(self._vectors)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()
        self._sparse.clear()
        self._payloads.clear()
