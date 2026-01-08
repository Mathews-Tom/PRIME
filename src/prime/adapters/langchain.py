"""LangChain adapter for PRIME.

Provides BaseRetriever implementation for LangChain ecosystem compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun


class PRIMERetriever(BaseRetriever):
    """LangChain retriever backed by PRIME.

    Integrates PRIME's predictive retrieval with LangChain's retriever interface,
    enabling use in LangChain chains and agents.

    Attributes:
        prime: PRIME instance for retrieval.
        mode: Retrieval mode - "process_turn" for contextual retrieval
            or "search" for direct memory search.
        session_id: Optional session ID for stateful conversations.
        top_k: Number of results to return (default: 5).

    Example:
        >>> from langchain.chains import RetrievalQA
        >>> from prime import PRIME, PRIMEConfig
        >>> prime = PRIME(PRIMEConfig.for_testing())
        >>> retriever = PRIMERetriever(prime=prime, mode="process_turn")
        >>> # Use in chain
        >>> qa_chain = RetrievalQA.from_chain_type(
        ...     llm=llm,
        ...     retriever=retriever,
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Use Any for prime to avoid Pydantic forward reference issues
    # Runtime type is prime.PRIME
    prime: Any = Field(description="PRIME instance for retrieval")
    mode: Literal["process_turn", "search"] = Field(
        default="process_turn",
        description="Retrieval mode",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for stateful conversations",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        """Get documents relevant to a query.

        Uses PRIME's process_turn for contextual retrieval or
        search_memory for direct search based on mode.

        Args:
            query: The query string.
            run_manager: Callback manager (unused).

        Returns:
            List of LangChain Documents with memory content.
        """
        if self.mode == "process_turn":
            result = self.prime.process_turn(
                content=query,
                session_id=self.session_id,
            )
            memories = result.memories
        else:
            memories = self.prime.search_memory(
                query=query,
                k=self.top_k,
            )

        return [
            Document(
                page_content=mem.content,
                metadata={
                    "similarity": mem.similarity,
                    "memory_id": mem.memory_id,
                    "cluster_id": mem.cluster_id,
                    **mem.metadata,
                },
            )
            for mem in memories
        ]
