"""Qdrant vector index implementation for MCS.

Provides the QdrantIndex class that implements the VectorIndex protocol
using Qdrant as the backend for both dense and sparse vector search.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from prime.mcs.exceptions import IndexError as MCSIndexError
from prime.mcs.index import IndexSearchResult, SparseVector
from prime.mcs.mcs_config import MCSConfig  # noqa: TC001 (needed at runtime)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class QdrantIndex:
    """Qdrant-backed vector index for MCS.

    Implements hybrid search with dense (semantic) and sparse (BM25)
    vectors using Qdrant's built-in RRF fusion.

    Attributes:
        config: MCS configuration with Qdrant settings.
        client: Qdrant client instance.
        collection_name: Name of the Qdrant collection.

    Example:
        >>> config = MCSConfig(qdrant_host="localhost", qdrant_port=6333)
        >>> index = QdrantIndex(config)
        >>> index.add("mem-1", embedding, sparse_vector, {"type": "preference"})
        >>> results = index.search_hybrid(query_embedding, sparse_query, top_k=5)
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "bm25"

    # UUID namespace for deterministic ID generation
    _UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    @classmethod
    def _id_to_uuid(cls, string_id: str) -> str:
        """Convert string ID to UUID for Qdrant storage.

        Uses UUID5 (SHA-1 based) for deterministic conversion,
        ensuring the same string ID always maps to the same UUID.

        Args:
            string_id: Original string identifier.

        Returns:
            UUID string representation.
        """
        return str(uuid.uuid5(cls._UUID_NAMESPACE, string_id))

    def __init__(
        self,
        config: MCSConfig,
        *,
        in_memory: bool = False,
    ) -> None:
        """Initialize QdrantIndex.

        Args:
            config: MCS configuration with Qdrant connection settings.
            in_memory: If True, use in-memory Qdrant (for testing).
        """
        self.config = config
        self.collection_name = config.collection_name

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port,
                api_key=config.qdrant_api_key,
                prefer_grpc=config.qdrant_prefer_grpc,
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self._create_collection()
                logger.info(
                    "Created Qdrant collection: %s",
                    self.collection_name,
                )
        except Exception as e:
            raise MCSIndexError(f"Failed to ensure collection: {e}") from e

    def _create_collection(self) -> None:
        """Create the Qdrant collection with hybrid search support."""
        vectors_config: dict[str, models.VectorParams] = {
            self.DENSE_VECTOR_NAME: models.VectorParams(
                size=self.config.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        }

        sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None
        if self.config.enable_hybrid_search:
            sparse_vectors_config = {
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

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

        Raises:
            MCSIndexError: If the operation fails.
        """
        try:
            vectors: dict[str, Any] = {
                self.DENSE_VECTOR_NAME: dense.tolist(),
            }

            if sparse is not None and self.config.enable_hybrid_search:
                vectors[self.SPARSE_VECTOR_NAME] = models.SparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                )

            # Store original ID in payload for reverse lookup
            full_payload: dict[str, str | int | float | bool] = {
                "_original_id": id,
                **(payload or {}),
            }

            # Convert string ID to UUID for Qdrant compatibility
            uuid_id = self._id_to_uuid(id)

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=uuid_id,
                        vector=vectors,
                        payload=full_payload,
                    ),
                ],
            )
        except Exception as e:
            raise MCSIndexError(f"Failed to add vector {id}: {e}") from e

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

        Raises:
            MCSIndexError: If the search fails.
        """
        try:
            query_filter = self._build_filter(filter_payload)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query.tolist(),
                using=self.DENSE_VECTOR_NAME,
                limit=top_k,
                query_filter=query_filter,
            )

            return [
                IndexSearchResult(
                    id=str(point.payload.get("_original_id", point.id))
                    if point.payload
                    else str(point.id),
                    score=point.score or 0.0,
                )
                for point in results.points
            ]
        except Exception as e:
            raise MCSIndexError(f"Dense search failed: {e}") from e

    def search_hybrid(
        self,
        dense_query: NDArray[np.float32],
        sparse_query: SparseVector,
        top_k: int,
        filter_payload: dict[str, str | int | float | bool] | None = None,
    ) -> list[IndexSearchResult]:
        """Search using hybrid dense + sparse with RRF fusion.

        Uses Qdrant's built-in RRF (Reciprocal Rank Fusion) to combine
        dense semantic search with sparse BM25 keyword search.

        Args:
            dense_query: Dense embedding query vector.
            sparse_query: Sparse vector for keyword matching.
            top_k: Number of results to return.
            filter_payload: Optional payload filter.

        Returns:
            List of search results with fused scores.

        Raises:
            MCSIndexError: If the search fails.
        """
        if not self.config.enable_hybrid_search:
            # Fall back to dense-only search
            return self.search_dense(query=dense_query, top_k=top_k, filter_payload=filter_payload)

        try:
            query_filter = self._build_filter(filter_payload)
            prefetch_limit = top_k * self.config.prefetch_multiplier

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query.tolist(),
                        using=self.DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query.indices,
                            values=sparse_query.values,
                        ),
                        using=self.SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
            )

            return [
                IndexSearchResult(
                    id=str(point.payload.get("_original_id", point.id))
                    if point.payload
                    else str(point.id),
                    score=point.score or 0.0,
                )
                for point in results.points
            ]
        except Exception as e:
            raise MCSIndexError(f"Hybrid search failed: {e}") from e

    def remove(self, id: str) -> bool:
        """Remove a vector from the index.

        Args:
            id: Identifier of the vector to remove.

        Returns:
            True if vector was removed, False if not found.

        Raises:
            MCSIndexError: If the operation fails.
        """
        try:
            uuid_id = self._id_to_uuid(id)

            # Check if point exists first
            existing = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[uuid_id],
            )

            if not existing:
                return False

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[uuid_id]),
            )
            return True
        except UnexpectedResponse:
            return False
        except Exception as e:
            raise MCSIndexError(f"Failed to remove vector {id}: {e}") from e

    def get(self, id: str) -> NDArray[np.float32] | None:
        """Get a vector by ID.

        Args:
            id: Identifier of the vector.

        Returns:
            The dense vector if found, None otherwise.

        Raises:
            MCSIndexError: If the operation fails.
        """
        import numpy as np

        try:
            uuid_id = self._id_to_uuid(id)

            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[uuid_id],
                with_vectors=[self.DENSE_VECTOR_NAME],
            )

            if not results:
                return None

            point = results[0]
            if point.vector is None:
                return None

            # Handle both dict and list vector formats
            if isinstance(point.vector, dict):
                dense_vector = point.vector.get(self.DENSE_VECTOR_NAME)
            else:
                dense_vector = point.vector

            if dense_vector is None:
                return None

            return np.array(dense_vector, dtype=np.float32)
        except UnexpectedResponse:
            return None
        except Exception as e:
            raise MCSIndexError(f"Failed to get vector {id}: {e}") from e

    def count(self) -> int:
        """Return the number of vectors in the index.

        Returns:
            Count of indexed vectors.

        Raises:
            MCSIndexError: If the operation fails.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            raise MCSIndexError(f"Failed to get count: {e}") from e

    def clear(self) -> None:
        """Remove all vectors from the index.

        Deletes and recreates the collection for a clean slate.

        Raises:
            MCSIndexError: If the operation fails.
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            logger.info("Cleared collection: %s", self.collection_name)
        except Exception as e:
            raise MCSIndexError(f"Failed to clear index: {e}") from e

    def _build_filter(
        self,
        filter_payload: dict[str, str | int | float | bool] | None,
    ) -> models.Filter | None:
        """Build Qdrant filter from payload conditions.

        Args:
            filter_payload: Key-value conditions for filtering.

        Returns:
            Qdrant Filter object or None if no filter.
        """
        if not filter_payload:
            return None

        # Build conditions list - cast value to acceptable MatchValue type
        conditions: list[
            models.FieldCondition
            | models.IsEmptyCondition
            | models.IsNullCondition
            | models.HasIdCondition
            | models.HasVectorCondition
            | models.NestedCondition
            | models.Filter
        ] = []
        for key, value in filter_payload.items():
            # MatchValue accepts bool | int | str, float needs conversion
            if isinstance(value, float):
                match_value: bool | int | str = str(value)
            else:
                match_value = value
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=match_value),
                )
            )

        return models.Filter(must=conditions)

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self.client.close()
