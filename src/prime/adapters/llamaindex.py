"""LlamaIndex adapter for PRIME.

Provides BaseRetriever implementation for LlamaIndex ecosystem compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

if TYPE_CHECKING:
    from prime import PRIME


class PRIMELlamaIndexRetriever(BaseRetriever):
    """LlamaIndex retriever backed by PRIME.

    Integrates PRIME's predictive retrieval with LlamaIndex's retriever interface,
    enabling use in LlamaIndex query engines and pipelines.

    Attributes:
        _prime: PRIME instance for retrieval.
        _mode: Retrieval mode - "process_turn" for contextual retrieval
            or "search" for direct memory search.
        _session_id: Optional session ID for stateful conversations.
        _top_k: Number of results to return (default: 5).

    Example:
        >>> from llama_index.core.query_engine import RetrieverQueryEngine
        >>> from prime import PRIME, PRIMEConfig
        >>> prime = PRIME(PRIMEConfig.for_testing())
        >>> retriever = PRIMELlamaIndexRetriever(prime=prime)
        >>> query_engine = RetrieverQueryEngine(retriever=retriever)
        >>> response = query_engine.query("What is JEPA?")
    """

    def __init__(
        self,
        prime: PRIME,
        mode: Literal["process_turn", "search"] = "process_turn",
        session_id: str | None = None,
        top_k: int = 5,
    ) -> None:
        """Initialize the retriever.

        Args:
            prime: PRIME instance for retrieval.
            mode: Retrieval mode - "process_turn" for contextual retrieval
                or "search" for direct memory search.
            session_id: Optional session ID for stateful conversations.
            top_k: Number of results to return (default: 5).
        """
        super().__init__()
        self._prime = prime
        self._mode = mode
        self._session_id = session_id
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes relevant to a query.

        Uses PRIME's process_turn for contextual retrieval or
        search_memory for direct search based on mode.

        Args:
            query_bundle: Query bundle containing the query string.

        Returns:
            List of NodeWithScore objects with memory content.
        """
        query = query_bundle.query_str

        if self._mode == "process_turn":
            result = self._prime.process_turn(
                content=query,
                session_id=self._session_id,
            )
            memories = result.memories
        else:
            memories = self._prime.search_memory(
                query=query,
                k=self._top_k,
            )

        return [
            NodeWithScore(
                node=TextNode(
                    text=mem.content,
                    metadata={
                        "memory_id": mem.memory_id,
                        "cluster_id": mem.cluster_id,
                        **mem.metadata,
                    },
                ),
                score=mem.similarity,
            )
            for mem in memories
        ]
