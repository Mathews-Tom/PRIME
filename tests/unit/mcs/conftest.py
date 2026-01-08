"""Shared fixtures for MCS unit tests.

Provides reusable test fixtures including MockEncoder, MockVectorIndex,
and pre-configured MCS instances for consistent testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from prime.mcs import MCSConfig, MemoryClusterStore, SparseVector
from prime.mcs.index import IndexSearchResult, VectorIndex


def _random_embedding(dim: int = 1024, seed: int | None = None) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


@dataclass
class MockEncoder:
    """Mock encoder for testing.

    Generates deterministic embeddings based on content hash.
    Supports preset embeddings for controlled test scenarios.
    """

    embedding_dim: int = 128
    max_length: int = 512
    model_name: str = "mock-encoder"
    _preset_embeddings: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Initialize preset embeddings dict."""
        if self._preset_embeddings is None:
            self._preset_embeddings = {}

    def encode(self, text: str) -> np.ndarray:
        """Encode text to deterministic embedding based on content hash."""
        if self._preset_embeddings and text in self._preset_embeddings:
            return self._preset_embeddings[text]
        seed = hash(text) % (2**31)
        return _random_embedding(self.embedding_dim, seed)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode batch of texts."""
        return [self.encode(text) for text in texts]

    def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Set a preset embedding for a specific text."""
        if self._preset_embeddings is None:
            self._preset_embeddings = {}
        self._preset_embeddings[text] = embedding

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "pooling_mode": "mean",
            "device": "cpu",
        }


class MockVectorIndex:
    """Mock vector index for testing.

    Stores vectors in memory and performs brute-force search.
    """

    def __init__(self) -> None:
        """Initialize mock index."""
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
        """Add a vector to the index."""
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
        """Search using dense vector similarity."""
        results: list[tuple[str, float]] = []

        for vec_id, vec in self._vectors.items():
            # Apply filter
            if filter_payload:
                payload = self._payloads.get(vec_id, {})
                if not all(payload.get(k) == v for k, v in filter_payload.items()):
                    continue

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
        """Search using hybrid dense + sparse (mock: just uses dense)."""
        del sparse_query  # Unused in mock - silence ARG002
        return self.search_dense(query=dense_query, top_k=top_k, filter_payload=filter_payload)

    def remove(self, id: str) -> bool:
        """Remove a vector from the index."""
        if id in self._vectors:
            del self._vectors[id]
            self._sparse.pop(id, None)
            self._payloads.pop(id, None)
            return True
        return False

    def get(self, id: str) -> np.ndarray | None:
        """Get a vector by ID."""
        return self._vectors.get(id)

    def count(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._vectors)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()
        self._sparse.clear()
        self._payloads.clear()


# Verify MockVectorIndex implements VectorIndex protocol
assert isinstance(MockVectorIndex(), VectorIndex)


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Create mock encoder with 128 dimensions."""
    return MockEncoder(embedding_dim=128)


@pytest.fixture
def mock_index() -> MockVectorIndex:
    """Create mock vector index."""
    return MockVectorIndex()


@pytest.fixture
def test_config() -> MCSConfig:
    """Create test configuration with small thresholds."""
    return MCSConfig(
        embedding_dim=128,
        similarity_threshold=0.85,
        consolidation_threshold=3,
        max_clusters=100,
        decay_factor=0.99,
        decay_unit_seconds=3600,
    )


@pytest.fixture
def mcs(
    mock_encoder: MockEncoder,
    mock_index: MockVectorIndex,
    test_config: MCSConfig,
) -> MemoryClusterStore:
    """Create MemoryClusterStore with mocks."""
    return MemoryClusterStore(
        encoder=mock_encoder,
        index=mock_index,
        config=test_config,
    )


@pytest.fixture
def similar_texts() -> list[str]:
    """Texts that should produce similar embeddings."""
    return [
        "User prefers dark mode",
        "User prefers dark mode theme",
        "User prefers dark mode setting",
    ]


@pytest.fixture
def dissimilar_texts() -> list[str]:
    """Texts that should produce dissimilar embeddings."""
    return [
        "User prefers dark mode",
        "The weather is sunny today",
        "Python is a programming language",
    ]
