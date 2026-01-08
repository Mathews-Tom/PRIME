"""Unit tests for framework adapters.

Tests LangChain and LlamaIndex adapter functionality with mocked PRIME.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prime import PRIMEConfig
from prime.adapters import PRIMELlamaIndexRetriever, PRIMERetriever
from prime.types import MemoryReadResult

if TYPE_CHECKING:
    pass


@pytest.fixture
def mock_prime() -> MagicMock:
    """Create mock PRIME instance."""
    prime = MagicMock()

    # Setup process_turn mock
    process_result = MagicMock()
    process_result.memories = [
        MemoryReadResult(
            memory_id="mem-1",
            content="Python is a programming language.",
            cluster_id=1,
            similarity=0.95,
            metadata={"source": "docs"},
            created_at=1000.0,
        ),
        MemoryReadResult(
            memory_id="mem-2",
            content="Python supports multiple paradigms.",
            cluster_id=1,
            similarity=0.88,
            metadata={"source": "wiki"},
            created_at=1001.0,
        ),
    ]
    prime.process_turn.return_value = process_result

    # Setup search_memory mock
    prime.search_memory.return_value = [
        MemoryReadResult(
            memory_id="mem-3",
            content="Python was created by Guido van Rossum.",
            cluster_id=2,
            similarity=0.85,
            metadata={"source": "history"},
            created_at=999.0,
        ),
    ]

    return prime


class TestPRIMERetriever:
    """Tests for LangChain PRIMERetriever."""

    def test_retriever_creation(self, mock_prime: MagicMock) -> None:
        """Test retriever can be created."""
        retriever = PRIMERetriever(prime=mock_prime)
        assert retriever.prime is mock_prime
        assert retriever.mode == "process_turn"
        assert retriever.session_id is None
        assert retriever.top_k == 5

    def test_retriever_with_options(self, mock_prime: MagicMock) -> None:
        """Test retriever with custom options."""
        retriever = PRIMERetriever(
            prime=mock_prime,
            mode="search",
            session_id="test-session",
            top_k=10,
        )
        assert retriever.mode == "search"
        assert retriever.session_id == "test-session"
        assert retriever.top_k == 10

    def test_get_relevant_documents_process_turn_mode(
        self,
        mock_prime: MagicMock,
    ) -> None:
        """Test retrieval in process_turn mode."""
        retriever = PRIMERetriever(
            prime=mock_prime,
            mode="process_turn",
            session_id="test-session",
        )

        # Create mock run_manager
        run_manager = MagicMock()

        docs = retriever._get_relevant_documents("What is Python?", run_manager=run_manager)

        mock_prime.process_turn.assert_called_once_with(
            content="What is Python?",
            session_id="test-session",
        )
        assert len(docs) == 2
        assert docs[0].page_content == "Python is a programming language."
        assert docs[0].metadata["similarity"] == 0.95
        assert docs[0].metadata["memory_id"] == "mem-1"
        assert docs[0].metadata["source"] == "docs"

    def test_get_relevant_documents_search_mode(
        self,
        mock_prime: MagicMock,
    ) -> None:
        """Test retrieval in search mode."""
        retriever = PRIMERetriever(
            prime=mock_prime,
            mode="search",
            top_k=5,
        )

        run_manager = MagicMock()
        docs = retriever._get_relevant_documents("Python history", run_manager=run_manager)

        mock_prime.search_memory.assert_called_once_with(
            query="Python history",
            k=5,
        )
        assert len(docs) == 1
        assert docs[0].page_content == "Python was created by Guido van Rossum."
        assert docs[0].metadata["similarity"] == 0.85

    def test_document_metadata_includes_memory_fields(
        self,
        mock_prime: MagicMock,
    ) -> None:
        """Test documents include all memory metadata."""
        retriever = PRIMERetriever(prime=mock_prime)
        run_manager = MagicMock()

        docs = retriever._get_relevant_documents("test", run_manager=run_manager)

        # Check required fields
        assert "memory_id" in docs[0].metadata
        assert "cluster_id" in docs[0].metadata
        assert "similarity" in docs[0].metadata
        # Check custom metadata merged
        assert "source" in docs[0].metadata


class TestPRIMELlamaIndexRetriever:
    """Tests for LlamaIndex PRIMELlamaIndexRetriever."""

    def test_retriever_creation(self, mock_prime: MagicMock) -> None:
        """Test retriever can be created."""
        retriever = PRIMELlamaIndexRetriever(prime=mock_prime)
        assert retriever._prime is mock_prime
        assert retriever._mode == "process_turn"
        assert retriever._session_id is None
        assert retriever._top_k == 5

    def test_retriever_with_options(self, mock_prime: MagicMock) -> None:
        """Test retriever with custom options."""
        retriever = PRIMELlamaIndexRetriever(
            prime=mock_prime,
            mode="search",
            session_id="test-session",
            top_k=10,
        )
        assert retriever._mode == "search"
        assert retriever._session_id == "test-session"
        assert retriever._top_k == 10

    def test_retrieve_process_turn_mode(
        self,
        mock_prime: MagicMock,
    ) -> None:
        """Test retrieval in process_turn mode."""
        from llama_index.core.schema import QueryBundle

        retriever = PRIMELlamaIndexRetriever(
            prime=mock_prime,
            mode="process_turn",
            session_id="test-session",
        )

        query_bundle = QueryBundle(query_str="What is Python?")
        nodes = retriever._retrieve(query_bundle)

        mock_prime.process_turn.assert_called_once_with(
            content="What is Python?",
            session_id="test-session",
        )
        assert len(nodes) == 2
        assert nodes[0].node.text == "Python is a programming language."
        assert nodes[0].score == 0.95
        assert nodes[0].node.metadata["memory_id"] == "mem-1"

    def test_retrieve_search_mode(
        self,
        mock_prime: MagicMock,
    ) -> None:
        """Test retrieval in search mode."""
        from llama_index.core.schema import QueryBundle

        retriever = PRIMELlamaIndexRetriever(
            prime=mock_prime,
            mode="search",
            top_k=5,
        )

        query_bundle = QueryBundle(query_str="Python history")
        nodes = retriever._retrieve(query_bundle)

        mock_prime.search_memory.assert_called_once_with(
            query="Python history",
            k=5,
        )
        assert len(nodes) == 1
        assert nodes[0].node.text == "Python was created by Guido van Rossum."
        assert nodes[0].score == 0.85

    def test_node_metadata_includes_memory_fields(
        self,
        mock_prime: MagicMock,
    ) -> None:
        """Test nodes include all memory metadata."""
        from llama_index.core.schema import QueryBundle

        retriever = PRIMELlamaIndexRetriever(prime=mock_prime)
        query_bundle = QueryBundle(query_str="test")

        nodes = retriever._retrieve(query_bundle)

        # Check required fields
        assert "memory_id" in nodes[0].node.metadata
        assert "cluster_id" in nodes[0].node.metadata
        # Check custom metadata merged
        assert "source" in nodes[0].node.metadata


class TestAdapterExports:
    """Tests for adapter module exports."""

    def test_langchain_retriever_import(self) -> None:
        """Test PRIMERetriever can be imported."""
        from prime.adapters import PRIMERetriever

        assert PRIMERetriever is not None

    def test_llamaindex_retriever_import(self) -> None:
        """Test PRIMELlamaIndexRetriever can be imported."""
        from prime.adapters import PRIMELlamaIndexRetriever

        assert PRIMELlamaIndexRetriever is not None

    def test_retrievers_from_adapters_module(self) -> None:
        """Test both retrievers from adapters module."""
        from prime import adapters

        assert hasattr(adapters, "PRIMERetriever")
        assert hasattr(adapters, "PRIMELlamaIndexRetriever")
