"""Protocol definitions for encoder interfaces.

Defines the Encoder protocol that both X-Encoder and Y-Encoder implement,
enabling polymorphic usage and backend swapping without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class Encoder(Protocol):
    """Protocol for embedding encoders.

    Defines the interface contract for both X-Encoder (query encoding)
    and Y-Encoder (content encoding). Implementations must provide all
    methods and properties defined here.

    This protocol enables:
    - Backend swapping (HuggingFace, SentenceTransformers, ONNX)
    - Testing with mock implementations
    - Type checking without concrete dependencies
    """

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension.

        Returns:
            Integer dimension of output embeddings.
        """
        ...

    @property
    def max_length(self) -> int:
        """Return maximum input sequence length in tokens.

        Returns:
            Maximum number of tokens before truncation.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier.

        Returns:
            HuggingFace model name or local path.
        """
        ...

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode single text to embedding vector.

        Args:
            text: Input text to encode. Must be non-empty.

        Returns:
            1D numpy array of shape (embedding_dim,) with float32 dtype.
            Output is L2-normalized if configured.

        Raises:
            EncodingError: If text is empty or encoding fails.
        """
        ...

    def encode_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Encode batch of texts to embedding vectors.

        Args:
            texts: List of input texts. Each must be non-empty.

        Returns:
            List of 1D numpy arrays, each of shape (embedding_dim,).

        Raises:
            EncodingError: If any text is empty or encoding fails.
        """
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata.

        Returns:
            Dictionary containing:
            - model_name: Model identifier
            - embedding_dim: Output dimension
            - max_length: Maximum sequence length
            - pooling_mode: Aggregation strategy
            - device: Compute device in use
        """
        ...
