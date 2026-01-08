"""Memory Cluster Store implementation.

Provides the main MemoryClusterStore class for intelligent memory management
with automatic clustering, consolidation, and temporal decay-weighted retrieval.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from prime.mcs.cluster import ClusterMember, MemoryCluster
from prime.mcs.exceptions import (
    ClusterNotFoundError,
    ConfigurationError,
    MCSError,
    SearchError,
    WriteError,
)
from prime.mcs.index import IndexSearchResult, SparseVector, VectorIndex
from prime.mcs.mcs_config import DEFAULT_CONFIG, MCSConfig
from prime.mcs.sparse import BM25Tokenizer
from prime.mcs.types import (
    ClusterInfo,
    ConsolidationResult,
    MemoryReadInput,
    MemoryReadResult,
    MemoryWriteInput,
    MemoryWriteResult,
    SearchMode,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prime.encoder.protocols import Encoder

logger = logging.getLogger(__name__)


class MemoryClusterStore:
    """Memory Cluster Store for intelligent memory management.

    Stores memories as embeddings with automatic consolidation of
    semantically similar content into cluster prototypes. Supports
    hybrid search (BM25 + vector) with RRF fusion for retrieval.

    Attributes:
        config: MCS configuration settings.
        encoder: Y-Encoder for content embedding.
        index: Vector index for similarity search.

    Example:
        >>> from prime.mcs import MemoryClusterStore, MCSConfig, QdrantIndex
        >>> from prime.encoder import YEncoder
        >>>
        >>> config = MCSConfig(similarity_threshold=0.85)
        >>> encoder = YEncoder()
        >>> index = QdrantIndex(config)
        >>> mcs = MemoryClusterStore(encoder=encoder, index=index, config=config)
        >>>
        >>> # Write a memory
        >>> result = mcs.write(MemoryWriteInput(content="User prefers dark mode"))
        >>> print(f"Stored in cluster {result.cluster_id}")
        >>>
        >>> # Search memories
        >>> query_embedding = encoder.encode("What are user preferences?")
        >>> results = mcs.read(MemoryReadInput(embedding=query_embedding.tolist(), k=5))
    """

    def __init__(
        self,
        encoder: Encoder,
        index: VectorIndex,
        config: MCSConfig | None = None,
    ) -> None:
        """Initialize MemoryClusterStore.

        Args:
            encoder: Y-Encoder for content embedding.
            index: Vector index for similarity search (e.g., QdrantIndex).
            config: MCS configuration. Defaults to DEFAULT_CONFIG.

        Raises:
            ConfigurationError: If encoder dimension doesn't match config.
        """
        self.config = config or DEFAULT_CONFIG
        self.encoder = encoder
        self.index = index

        # Validate encoder dimension matches config
        if encoder.embedding_dim != self.config.embedding_dim:
            raise ConfigurationError(
                f"Encoder embedding_dim ({encoder.embedding_dim}) doesn't match "
                f"config embedding_dim ({self.config.embedding_dim})"
            )

        # In-memory cluster storage: cluster_id -> MemoryCluster
        self._clusters: dict[int, MemoryCluster] = {}
        self._next_cluster_id: int = 0

        # Memory ID to cluster ID mapping for lookups
        self._memory_to_cluster: dict[str, int] = {}

        # BM25 tokenizer for sparse vectors
        self._tokenizer = BM25Tokenizer(
            min_token_length=2,
            max_vocab_size=self.config.sparse_vocab_size,
        )

        logger.info(
            "Initialized MCS with embedding_dim=%d, similarity_threshold=%.2f",
            self.config.embedding_dim,
            self.config.similarity_threshold,
        )

    @property
    def cluster_count(self) -> int:
        """Return number of clusters in the store."""
        return len(self._clusters)

    @property
    def memory_count(self) -> int:
        """Return total number of memories across all clusters."""
        return sum(cluster.size for cluster in self._clusters.values())

    def write(self, input_data: MemoryWriteInput) -> MemoryWriteResult:
        """Write content to memory store.

        Encodes content, assigns to appropriate cluster (existing or new),
        and triggers consolidation if threshold is reached.

        Args:
            input_data: Write input with content and metadata.

        Returns:
            MemoryWriteResult with cluster assignment details.

        Raises:
            WriteError: If encoding or storage fails.
            ConfigurationError: If max clusters exceeded.
        """
        try:
            # Encode content to embedding
            embedding = self.encoder.encode(input_data.content)

            # Build payload for index
            payload = self._build_payload(input_data)

            # Find or create cluster
            if input_data.force_new_cluster:
                cluster, is_new = self._create_cluster(), True
                similarity = 1.0
            else:
                cluster, is_new, similarity = self._assign_to_cluster(embedding)

            # Generate memory ID and add to cluster
            memory_id = str(uuid.uuid4())
            cluster.add_member(
                embedding=embedding,
                content=input_data.content,
                metadata=payload,
                memory_id=memory_id,
            )

            # Track memory to cluster mapping
            self._memory_to_cluster[memory_id] = cluster.cluster_id

            # Update BM25 tokenizer with new document for IDF calculation
            self._tokenizer.fit_incremental(input_data.content)

            # Add to vector index for retrieval
            sparse_vector = self._compute_sparse_vector(input_data.content)
            self.index.add(
                id=memory_id,
                dense=embedding,
                sparse=sparse_vector,
                payload={"cluster_id": cluster.cluster_id, **payload},
            )

            # Check if consolidation needed
            consolidated = False
            if cluster.size >= self.config.consolidation_threshold:
                consolidated = self._consolidate_cluster(cluster)

            logger.debug(
                "Wrote memory %s to cluster %d (new=%s, consolidated=%s)",
                memory_id,
                cluster.cluster_id,
                is_new,
                consolidated,
            )

            return MemoryWriteResult(
                memory_id=memory_id,
                cluster_id=cluster.cluster_id,
                is_new_cluster=is_new,
                consolidated=consolidated,
                similarity_to_prototype=similarity,
            )

        except MCSError:
            raise
        except Exception as e:
            raise WriteError(f"Failed to write memory: {e}") from e

    def read(self, input_data: MemoryReadInput) -> list[MemoryReadResult]:
        """Search memories by embedding similarity.

        Performs dense, sparse, or hybrid search with temporal decay
        weighting applied to results.

        Args:
            input_data: Read input with query embedding and parameters.

        Returns:
            List of MemoryReadResult ordered by decay-adjusted score.

        Raises:
            SearchError: If search operation fails.
        """
        try:
            query_embedding = np.array(input_data.embedding, dtype=np.float32)

            # Build filter for user/session scoping
            filter_payload = self._build_filter(input_data)

            # Perform search based on mode
            if input_data.search_mode == SearchMode.DENSE:
                search_results = self.index.search_dense(
                    query=query_embedding,
                    top_k=input_data.k,
                    filter_payload=filter_payload,
                )
            elif input_data.search_mode == SearchMode.HYBRID:
                sparse_query = self._compute_sparse_vector(
                    input_data.query_text or "", is_query=True
                )
                search_results = self.index.search_hybrid(
                    dense_query=query_embedding,
                    sparse_query=sparse_query,
                    top_k=input_data.k,
                    filter_payload=filter_payload,
                )
            else:  # SPARSE - fall back to dense for now
                search_results = self.index.search_dense(
                    query=query_embedding,
                    top_k=input_data.k,
                    filter_payload=filter_payload,
                )

            # Convert search results to MemoryReadResults with decay
            results = self._process_search_results(
                search_results,
                input_data.min_similarity,
            )

            return results

        except MCSError:
            raise
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

    def get_cluster(self, cluster_id: int) -> ClusterInfo:
        """Get information about a specific cluster.

        Args:
            cluster_id: ID of the cluster to retrieve.

        Returns:
            ClusterInfo with cluster statistics.

        Raises:
            ClusterNotFoundError: If cluster doesn't exist.
        """
        if cluster_id not in self._clusters:
            raise ClusterNotFoundError(cluster_id)

        cluster = self._clusters[cluster_id]
        cluster.touch()

        return ClusterInfo(
            cluster_id=cluster.cluster_id,
            size=cluster.size,
            is_consolidated=cluster.is_consolidated,
            prototype_norm=float(np.linalg.norm(cluster.prototype)),
            creation_timestamp=cluster.created_at,
            last_access_timestamp=cluster.last_accessed_at,
            access_count=cluster.access_count,
            representative_content=cluster.representative_content[:200],
        )

    def get_cluster_info(self, cluster_id: int) -> ClusterInfo | None:
        """Get information about a specific cluster (returns None if not found).

        Unlike get_cluster(), this method returns None instead of raising
        an exception when the cluster doesn't exist.

        Args:
            cluster_id: ID of the cluster to retrieve.

        Returns:
            ClusterInfo if found, None otherwise.
        """
        if cluster_id not in self._clusters:
            return None

        cluster = self._clusters[cluster_id]
        cluster.touch()

        return ClusterInfo(
            cluster_id=cluster.cluster_id,
            size=cluster.size,
            is_consolidated=cluster.is_consolidated,
            prototype_norm=float(np.linalg.norm(cluster.prototype)),
            creation_timestamp=cluster.created_at,
            last_access_timestamp=cluster.last_accessed_at,
            access_count=cluster.access_count,
            representative_content=cluster.representative_content[:200],
        )

    def list_clusters(self) -> list[ClusterInfo]:
        """List all clusters in the store.

        Returns:
            List of ClusterInfo for all clusters.
        """
        result: list[ClusterInfo] = []

        for cluster in self._clusters.values():
            result.append(
                ClusterInfo(
                    cluster_id=cluster.cluster_id,
                    size=cluster.size,
                    is_consolidated=cluster.is_consolidated,
                    prototype_norm=float(np.linalg.norm(cluster.prototype)),
                    creation_timestamp=cluster.created_at,
                    last_access_timestamp=cluster.last_accessed_at,
                    access_count=cluster.access_count,
                    representative_content=cluster.representative_content[:200],
                )
            )

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get overall store statistics.

        Returns:
            Dictionary with store metrics:
            - cluster_count: Number of clusters
            - memory_count: Total memories
            - consolidated_clusters: Number of consolidated clusters
            - avg_cluster_size: Average members per cluster
            - compression_ratio: Memories / clusters ratio
        """
        cluster_count = self.cluster_count
        memory_count = self.memory_count
        consolidated = sum(
            1 for c in self._clusters.values() if c.is_consolidated
        )

        avg_size = memory_count / cluster_count if cluster_count > 0 else 0.0
        compression = memory_count / cluster_count if cluster_count > 0 else 0.0

        return {
            "cluster_count": cluster_count,
            "memory_count": memory_count,
            "consolidated_clusters": consolidated,
            "avg_cluster_size": avg_size,
            "compression_ratio": compression,
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "similarity_threshold": self.config.similarity_threshold,
                "consolidation_threshold": self.config.consolidation_threshold,
                "max_clusters": self.config.max_clusters,
                "enable_hybrid_search": self.config.enable_hybrid_search,
            },
        }

    def consolidate_all(self) -> ConsolidationResult:
        """Consolidate all clusters that exceed threshold.

        Returns:
            ConsolidationResult with statistics.
        """
        start_time = time.time()
        clusters_processed = 0
        memories_consolidated = 0

        for cluster in self._clusters.values():
            if (
                not cluster.is_consolidated
                and cluster.size >= self.config.consolidation_threshold
            ):
                members_removed = cluster.consolidate()
                if members_removed > 0:
                    clusters_processed += 1
                    memories_consolidated += members_removed

        duration_ms = (time.time() - start_time) * 1000
        storage_reduction = (
            (memories_consolidated / (memories_consolidated + self.memory_count)) * 100
            if memories_consolidated > 0
            else 0.0
        )

        logger.info(
            "Consolidated %d clusters, removed %d memories in %.2fms",
            clusters_processed,
            memories_consolidated,
            duration_ms,
        )

        return ConsolidationResult(
            clusters_processed=clusters_processed,
            memories_consolidated=memories_consolidated,
            storage_reduction=storage_reduction,
            duration_ms=duration_ms,
        )

    def _assign_to_cluster(
        self,
        embedding: NDArray[np.float32],
    ) -> tuple[MemoryCluster, bool, float]:
        """Assign embedding to existing or new cluster.

        Args:
            embedding: L2-normalized embedding vector.

        Returns:
            Tuple of (cluster, is_new_cluster, similarity_to_prototype).

        Raises:
            ConfigurationError: If max clusters exceeded.
        """
        if not self._clusters:
            return self._create_cluster(), True, 1.0

        # Find nearest cluster
        best_cluster: MemoryCluster | None = None
        best_similarity = 0.0

        for cluster in self._clusters.values():
            if cluster.is_empty:
                continue
            similarity = cluster.similarity_to(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        # Check if similarity exceeds threshold
        if best_cluster is not None and best_similarity >= self.config.similarity_threshold:
            return best_cluster, False, best_similarity

        # Create new cluster
        return self._create_cluster(), True, 1.0

    def _create_cluster(self) -> MemoryCluster:
        """Create a new empty cluster.

        Returns:
            New MemoryCluster instance.

        Raises:
            ConfigurationError: If max clusters exceeded.
        """
        if len(self._clusters) >= self.config.max_clusters:
            raise ConfigurationError(
                f"Maximum cluster count ({self.config.max_clusters}) exceeded"
            )

        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1

        cluster = MemoryCluster.create(
            cluster_id=cluster_id,
            embedding_dim=self.config.embedding_dim,
        )
        self._clusters[cluster_id] = cluster

        logger.debug("Created new cluster %d", cluster_id)
        return cluster

    def _consolidate_cluster(self, cluster: MemoryCluster) -> bool:
        """Consolidate a cluster if not already consolidated.

        Args:
            cluster: Cluster to consolidate.

        Returns:
            True if consolidation was performed.
        """
        if cluster.is_consolidated:
            return False

        members_removed = cluster.consolidate()
        if members_removed > 0:
            logger.debug(
                "Consolidated cluster %d, removed %d members",
                cluster.cluster_id,
                members_removed,
            )
            return True
        return False

    def _build_payload(
        self,
        input_data: MemoryWriteInput,
    ) -> dict[str, str | int | float | bool]:
        """Build payload dictionary from write input.

        Args:
            input_data: Write input with metadata.

        Returns:
            Payload dictionary for storage.
        """
        payload: dict[str, str | int | float | bool] = {}

        if input_data.metadata:
            payload.update(input_data.metadata)

        if input_data.user_id:
            payload["user_id"] = input_data.user_id

        if input_data.session_id:
            payload["session_id"] = input_data.session_id

        return payload

    def _build_filter(
        self,
        input_data: MemoryReadInput,
    ) -> dict[str, str | int | float | bool] | None:
        """Build filter dictionary from read input.

        Args:
            input_data: Read input with filter parameters.

        Returns:
            Filter dictionary or None if no filters.
        """
        filters: dict[str, str | int | float | bool] = {}

        if input_data.user_id:
            filters["user_id"] = input_data.user_id

        if input_data.session_id:
            filters["session_id"] = input_data.session_id

        return filters if filters else None

    def _compute_sparse_vector(self, text: str, *, is_query: bool = False) -> SparseVector:
        """Compute sparse vector for BM25 search using trained tokenizer.

        Uses BM25Tokenizer with IDF weights for proper term weighting.

        Args:
            text: Input text to tokenize.
            is_query: If True, use query-optimized encoding.

        Returns:
            SparseVector with BM25-weighted term indices and values.
        """
        if not text:
            return SparseVector(indices=[], values=[])

        if is_query:
            return self._tokenizer.encode_query(text)
        return self._tokenizer.encode(text)

    def _process_search_results(
        self,
        search_results: list[IndexSearchResult],
        min_similarity: float,
    ) -> list[MemoryReadResult]:
        """Process raw search results into MemoryReadResults.

        Applies temporal decay and filters by minimum similarity.

        Args:
            search_results: Raw results from index search.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of MemoryReadResult with decay-adjusted scores.
        """
        results: list[MemoryReadResult] = []
        current_time = time.time()

        for rank, result in enumerate(search_results, start=1):
            memory_id = result.id
            similarity = result.score

            # Skip if below minimum similarity
            if similarity < min_similarity:
                continue

            # Find cluster and member
            cluster_id = self._memory_to_cluster.get(memory_id)
            if cluster_id is None:
                continue

            cluster = self._clusters.get(cluster_id)
            if cluster is None:
                continue

            member = cluster.get_member_by_id(memory_id)
            if member is None:
                continue

            # Apply temporal decay
            age_seconds = current_time - member.created_at
            age_units = age_seconds / self.config.decay_unit_seconds
            decay_adjusted = similarity * (self.config.decay_factor ** age_units)

            # Check if this is the representative
            is_representative = (
                cluster.representative_member.memory_id == memory_id
            )

            results.append(
                MemoryReadResult(
                    memory_id=memory_id,
                    content=member.content,
                    embedding=member.embedding.tolist(),
                    metadata=dict(member.metadata),
                    cluster_id=cluster_id,
                    similarity=similarity,
                    decay_adjusted_score=decay_adjusted,
                    is_representative=is_representative,
                    rank=rank,
                )
            )

        # Re-sort by decay-adjusted score and update ranks
        results.sort(key=lambda r: r.decay_adjusted_score, reverse=True)
        for idx, read_result in enumerate(results):
            # Create new result with updated rank (immutable model)
            results[idx] = MemoryReadResult(
                memory_id=read_result.memory_id,
                content=read_result.content,
                embedding=read_result.embedding,
                metadata=read_result.metadata,
                cluster_id=read_result.cluster_id,
                similarity=read_result.similarity,
                decay_adjusted_score=read_result.decay_adjusted_score,
                is_representative=read_result.is_representative,
                rank=idx + 1,
            )

        return results

    def _get_member_from_cluster(
        self,
        cluster: MemoryCluster,
        memory_id: str,
    ) -> ClusterMember | None:
        """Get member from cluster by memory_id.

        Args:
            cluster: Cluster to search in.
            memory_id: Memory ID to find.

        Returns:
            ClusterMember if found, None otherwise.
        """
        return cluster.get_member_by_id(memory_id)
