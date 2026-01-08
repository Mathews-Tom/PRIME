"""Pytest fixtures for SSM unit tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from prime.ssm import SSMConfig, SemanticStateMonitor


class MockEncoder:
    """Mock X-Encoder for isolated SSM testing.

    Implements the Encoder protocol with deterministic embeddings
    for reproducible tests.
    """

    def __init__(self, embedding_dim: int = 1024) -> None:
        """Initialize mock encoder.

        Args:
            embedding_dim: Dimension of embeddings to generate.
        """
        self._embedding_dim = embedding_dim
        self._embeddings: list[NDArray[np.float32]] = []
        self._call_count = 0

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    @property
    def max_length(self) -> int:
        """Return max input length."""
        return 512

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "mock-encoder"

    def encode(self, text: str) -> NDArray[np.float32]:
        """Return predetermined or deterministic embedding.

        If embeddings were set via set_embeddings(), returns them in order.
        Otherwise, generates a deterministic embedding based on text hash.

        Args:
            text: Text to encode.

        Returns:
            Embedding vector.
        """
        self._call_count += 1
        if self._embeddings:
            return self._embeddings.pop(0)
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(self._embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)  # Normalize

    def encode_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Encode batch of texts.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors.
        """
        return [self.encode(t) for t in texts]

    def get_model_info(self) -> dict[str, Any]:
        """Return model info."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "max_length": self.max_length,
        }

    def set_embeddings(self, embeddings: list[NDArray[np.float32]]) -> None:
        """Set predetermined embeddings for testing.

        Args:
            embeddings: List of embeddings to return in order.
        """
        self._embeddings = list(embeddings)

    @property
    def call_count(self) -> int:
        """Return number of encode calls."""
        return self._call_count


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Create mock encoder with default 1024 dimensions."""
    return MockEncoder(embedding_dim=1024)


@pytest.fixture
def ssm_config() -> SSMConfig:
    """Create default test SSM configuration."""
    return SSMConfig(
        window_size=5,
        variance_threshold=0.15,
        smoothing_factor=0.3,
        prepare_ratio=0.5,
        consolidate_ratio=2.0,
        embedding_dim=1024,
    )


@pytest.fixture
def ssm(mock_encoder: MockEncoder, ssm_config: SSMConfig) -> SemanticStateMonitor:
    """Create SSM instance with mock encoder and default config."""
    return SemanticStateMonitor(encoder=mock_encoder, config=ssm_config)
