"""PRIME orchestration class.

Integrates SSM, Predictor, MCS, and Y-Encoder into a unified
predictive retrieval-augmented generation system.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from prime.encoder import YEncoder
from prime.exceptions import ComponentError
from prime.mcs import MemoryClusterStore, MemoryIndex, QdrantIndex
from prime.mcs.types import MemoryReadInput, MemoryWriteInput
from prime.mcs.types import MemoryReadResult as MCSReadResult
from prime.mcs.types import MemoryWriteResult as MCSWriteResult
from prime.predictor import EmbeddingPredictor
from prime.ssm import SemanticStateMonitor
from prime.types import (
    ActionState,
    ComponentStatus,
    MemoryReadResult,
    MemoryWriteResult,
    PRIMEDiagnostics,
    PRIMEResponse,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prime.config import PRIMEConfig

logger = structlog.get_logger(__name__)


class PRIME:
    """Main PRIME orchestration class.

    Integrates SSM, Predictor, MCS, and Y-Encoder into a unified
    retrieval-augmented generation system that predicts what context
    is needed rather than reactively searching.

    The PRIME class implements predictive retrieval based on Meta FAIR's
    VL-JEPA architecture, using variance-based semantic boundary detection
    to trigger intelligent context retrieval.

    Example:
        >>> config = PRIMEConfig.for_testing()
        >>> prime = PRIME(config)
        >>>
        >>> # Process a conversation turn
        >>> response = prime.process_turn(
        ...     "What is machine learning?",
        ...     session_id="sess_123",
        ... )
        >>>
        >>> # Record an LLM response for future retrieval
        >>> prime.record_response(
        ...     "Machine learning is a subset of AI...",
        ...     session_id="sess_123",
        ... )

    Attributes:
        config: PRIME configuration instance.
        y_encoder: Y-Encoder for text to embedding conversion.
        ssm: Semantic State Monitor for boundary detection.
        mcs: Memory Cluster Store for memory storage/retrieval.
        predictor: Embedding Predictor for target embedding prediction.
    """

    __slots__ = (
        "_error_count",
        "_request_count",
        "_session_context",
        "_start_time",
        "_version",
        "config",
        "mcs",
        "predictor",
        "ssm",
        "y_encoder",
    )

    def __init__(self, config: PRIMEConfig) -> None:
        """Initialize PRIME with all components.

        Args:
            config: PRIME configuration with component settings.

        Raises:
            ComponentError: If any component fails to initialize.
        """
        self.config = config
        self._start_time = time.time()
        self._version = "1.0.0"
        self._session_context: dict[str, deque[NDArray[np.float32]]] = {}
        self._request_count = 0
        self._error_count = 0

        self._init_components()

        logger.info(
            "prime_initialized",
            version=self._version,
            embedding_dim=config.y_encoder.embedding_dim,
        )

    def _init_components(self) -> None:
        """Initialize all PRIME components.

        Raises:
            ComponentError: If component initialization fails.
        """
        try:
            self.y_encoder = YEncoder(self.config.y_encoder)
            logger.debug("y_encoder_initialized")
        except Exception as e:
            raise ComponentError("y_encoder", str(e)) from e

        try:
            self.ssm = SemanticStateMonitor(
                config=self.config.ssm,
                encoder=self.y_encoder,
            )
            logger.debug("ssm_initialized")
        except Exception as e:
            raise ComponentError("ssm", str(e)) from e

        try:
            # Create vector index based on config
            if self.config.mcs.index_type == "faiss":
                index = MemoryIndex(
                    embedding_dim=self.config.y_encoder.embedding_dim
                )
            else:
                index = QdrantIndex(config=self.config.mcs)
            self.mcs = MemoryClusterStore(
                encoder=self.y_encoder,
                index=index,
                config=self.config.mcs,
            )
            logger.debug("mcs_initialized")
        except Exception as e:
            raise ComponentError("mcs", str(e)) from e

        try:
            self.predictor = EmbeddingPredictor(self.config.predictor)
            logger.debug("predictor_initialized")
        except Exception as e:
            raise ComponentError("predictor", str(e)) from e

    def process_turn(
        self,
        input_text: str,
        *,
        session_id: str | None = None,
        force_retrieval: bool = False,
        k: int = 5,
    ) -> PRIMEResponse:
        """Process a conversation turn through the PRIME pipeline.

        This is the main entry point for predictive RAG processing. It:
        1. Generates a session_id if not provided
        2. Updates SSM state and checks for semantic boundary crossing
        3. If boundary crossed or force_retrieval: predicts target embedding
        4. Searches MCS with predicted embedding for relevant memories
        5. Triggers consolidation if action is RETRIEVE_CONSOLIDATE

        Args:
            input_text: User input text to process.
            session_id: Optional session identifier. If None, a new UUID is generated.
            force_retrieval: If True, bypass SSM boundary detection and force retrieval.
            k: Number of memories to retrieve (default: 5).

        Returns:
            PRIMEResponse with retrieved memories, boundary detection info,
            variance metrics, and processing latency.

        Raises:
            ComponentError: If any component operation fails.
        """
        start = time.perf_counter()
        self._request_count += 1

        # Generate session_id if not provided
        if session_id is None:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"

        log = logger.bind(session_id=session_id)
        log.info(
            "process_turn_start",
            input_length=len(input_text),
            force_retrieval=force_retrieval,
        )

        try:
            # Step 1: Update SSM (encodes internally)
            ssm_result = self.ssm.update(input_text)

            # Store embedding in session context for predictor
            self._update_session_context(session_id, ssm_result.embedding)

            # Step 2: Decide if retrieval is needed
            should_retrieve = force_retrieval or ssm_result.action in (
                ActionState.RETRIEVE,
                ActionState.RETRIEVE_CONSOLIDATE,
            )

            retrieved_memories: list[MemoryReadResult] = []

            if should_retrieve:
                # Step 3a: Get context window for predictor
                context_embeddings = self._get_context_window(session_id)

                # Convert current embedding to numpy array
                current_embedding = np.array(
                    ssm_result.embedding, dtype=np.float32
                )

                # Step 3b: Predict target embedding
                predicted_embedding = self.predictor.predict(
                    context_embeddings=context_embeddings,
                    query_embedding=current_embedding,
                )

                # Step 3c: Search MCS with predicted embedding
                search_input = MemoryReadInput(
                    embedding=predicted_embedding.tolist(),
                    query_text=input_text,
                    k=k,
                    session_id=session_id,
                )
                mcs_results = self.mcs.read(search_input)

                # Convert MCS results to API types
                retrieved_memories = [
                    self._convert_mcs_read_result(r) for r in mcs_results
                ]

                log.info(
                    "retrieval_complete",
                    memory_count=len(retrieved_memories),
                )

            # Step 4: Consolidate if needed
            if ssm_result.action == ActionState.RETRIEVE_CONSOLIDATE:
                consolidation_result = self.mcs.consolidate_all()
                log.info(
                    "consolidation_triggered",
                    clusters_processed=consolidation_result.clusters_processed,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            log.info(
                "process_turn_complete",
                action=ssm_result.action.value,
                boundary_crossed=ssm_result.boundary_crossed,
                latency_ms=latency_ms,
            )

            return PRIMEResponse(
                retrieved_memories=retrieved_memories,
                boundary_crossed=ssm_result.boundary_crossed,
                variance=ssm_result.variance,
                smoothed_variance=ssm_result.smoothed_variance,
                action=ssm_result.action,
                session_id=session_id,
                turn_number=ssm_result.turn_number,
                latency_ms=latency_ms,
            )

        except Exception as e:
            self._error_count += 1
            log.error("process_turn_failed", error=str(e))
            raise ComponentError("prime", f"process_turn failed: {e}") from e

    def record_response(
        self,
        content: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Record an LLM response to memory for future retrieval.

        Call this after generating an LLM response to store it in
        PRIME memory. The response is encoded and assigned to an
        appropriate cluster.

        Args:
            content: LLM response text to store.
            session_id: Optional session identifier for grouping.
            metadata: Optional metadata dictionary for filtering.

        Returns:
            MemoryWriteResult with memory_id and cluster assignment.

        Raises:
            ComponentError: If write operation fails.
        """
        logger.info(
            "record_response",
            session_id=session_id,
            content_length=len(content),
        )

        write_input = MemoryWriteInput(
            content=content,
            session_id=session_id,
            metadata=metadata,
        )

        mcs_result = self.mcs.write(write_input)
        return self._convert_mcs_write_result(mcs_result)

    def write_external_knowledge(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Write external knowledge to memory for document ingestion.

        Use this for adding knowledge that doesn't come from conversation,
        such as documents, articles, or pre-existing knowledge bases.

        Args:
            content: Knowledge content to store.
            metadata: Optional metadata dictionary. Will include source="external"
                by default.

        Returns:
            MemoryWriteResult with memory_id and cluster assignment.

        Raises:
            ComponentError: If write operation fails.
        """
        logger.info(
            "write_external_knowledge",
            content_length=len(content),
        )

        # Add source to metadata
        final_metadata: dict[str, str | int | float | bool] = {
            "source": "external"
        }
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    final_metadata[key] = value

        write_input = MemoryWriteInput(
            content=content,
            metadata=final_metadata,
        )

        mcs_result = self.mcs.write(write_input)
        return self._convert_mcs_write_result(mcs_result)

    def search_memory(
        self,
        query: str,
        *,
        k: int = 5,
        min_similarity: float = 0.0,
        session_id: str | None = None,
    ) -> list[MemoryReadResult]:
        """Direct memory search without SSM boundary detection.

        Use this for explicit search operations where you want to bypass
        the predictive retrieval mechanism and directly query memories.

        Args:
            query: Query text for searching memories.
            k: Number of results to return (default: 5).
            min_similarity: Minimum similarity threshold (default: 0.0).
            session_id: Optional session filter for scoping results.

        Returns:
            List of MemoryReadResult matching the query.
        """
        logger.info(
            "search_memory",
            query_length=len(query),
            k=k,
        )

        # Encode query
        query_embedding = self.y_encoder.encode(query)

        search_input = MemoryReadInput(
            embedding=query_embedding.tolist(),
            query_text=query,
            k=k,
            min_similarity=min_similarity,
            session_id=session_id,
        )

        mcs_results = self.mcs.read(search_input)
        return [self._convert_mcs_read_result(r) for r in mcs_results]

    def get_diagnostics(self) -> PRIMEDiagnostics:
        """Get comprehensive system diagnostics.

        Returns health status, performance metrics, and component
        information for monitoring and debugging.

        Returns:
            PRIMEDiagnostics with status, uptime, components, and metrics.
        """
        uptime = time.time() - self._start_time
        error_rate = (
            self._error_count / self._request_count
            if self._request_count > 0
            else 0.0
        )

        # Get component stats
        ssm_state = self.ssm.get_state()
        mcs_stats = self.mcs.get_stats()

        # Build component status
        components = {
            "ssm": ComponentStatus(
                name="SemanticStateMonitor",
                status="healthy",
                latency_p50_ms=0.0,  # Could be instrumented
                error_rate=0.0,
            ),
            "mcs": ComponentStatus(
                name="MemoryClusterStore",
                status="healthy",
                latency_p50_ms=0.0,
                error_rate=0.0,
            ),
            "predictor": ComponentStatus(
                name="EmbeddingPredictor",
                status="healthy",
                latency_p50_ms=0.0,
                error_rate=0.0,
            ),
            "y_encoder": ComponentStatus(
                name="YEncoder",
                status="healthy",
                latency_p50_ms=0.0,
                error_rate=0.0,
            ),
        }

        # Determine overall status based on error rate
        if error_rate > 0.1:
            status = "unhealthy"
        elif error_rate > 0.01:
            status = "degraded"
        else:
            status = "healthy"

        return PRIMEDiagnostics(
            status=status,
            version=self._version,
            uptime_seconds=uptime,
            components=components,
            metrics={
                "total_requests": float(self._request_count),
                "total_errors": float(self._error_count),
                "error_rate": error_rate,
                "active_sessions": float(len(self._session_context)),
                "ssm_turn_number": float(ssm_state.get("turn_number", 0)),
                "mcs_cluster_count": float(mcs_stats.get("cluster_count", 0)),
                "mcs_memory_count": float(mcs_stats.get("memory_count", 0)),
            },
        )

    def reset_session(self, session_id: str) -> None:
        """Reset session state.

        Clears the context window for the specified session and resets
        the SSM state. Call this when starting a new conversation.

        Args:
            session_id: Session identifier to reset.
        """
        logger.info("reset_session", session_id=session_id)

        if session_id in self._session_context:
            del self._session_context[session_id]

        self.ssm.reset()

    def _update_session_context(
        self,
        session_id: str,
        embedding: list[float],
    ) -> None:
        """Update session context with new embedding.

        Args:
            session_id: Session identifier.
            embedding: Embedding vector as list of floats.
        """
        if session_id not in self._session_context:
            # Use SSM window size as context buffer size
            self._session_context[session_id] = deque(
                maxlen=self.config.ssm.window_size
            )

        embedding_array = np.array(embedding, dtype=np.float32)
        self._session_context[session_id].append(embedding_array)

    def _get_context_window(
        self,
        session_id: str,
    ) -> NDArray[np.float32]:
        """Get context window for predictor.

        Args:
            session_id: Session identifier.

        Returns:
            Context embeddings as (N, D) numpy array.
        """
        if session_id not in self._session_context:
            # Return empty context
            return np.zeros(
                (1, self.config.y_encoder.embedding_dim),
                dtype=np.float32,
            )

        context = self._session_context[session_id]
        if len(context) == 0:
            return np.zeros(
                (1, self.config.y_encoder.embedding_dim),
                dtype=np.float32,
            )

        return np.stack(list(context), axis=0)

    @staticmethod
    def _convert_mcs_read_result(mcs_result: MCSReadResult) -> MemoryReadResult:
        """Convert MCS read result to API type.

        Args:
            mcs_result: MCS-specific read result.

        Returns:
            API MemoryReadResult dataclass.
        """
        return MemoryReadResult(
            memory_id=mcs_result.memory_id,
            content=mcs_result.content,
            cluster_id=mcs_result.cluster_id,
            similarity=mcs_result.similarity,
            metadata=mcs_result.metadata,
            created_at=0.0,  # MCS doesn't expose this directly
        )

    @staticmethod
    def _convert_mcs_write_result(
        mcs_result: MCSWriteResult,
    ) -> MemoryWriteResult:
        """Convert MCS write result to API type.

        Args:
            mcs_result: MCS-specific write result.

        Returns:
            API MemoryWriteResult dataclass.
        """
        return MemoryWriteResult(
            memory_id=mcs_result.memory_id,
            cluster_id=mcs_result.cluster_id,
            is_new_cluster=mcs_result.is_new_cluster,
            consolidated=mcs_result.consolidated,
        )
