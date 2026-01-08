"""Y-Encoder implementation for target content embedding.

Encodes text content into L2-normalized embeddings for memory storage
and predictor training targets.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from prime.encoder.config import YEncoderConfig
from prime.encoder.exceptions import EncodingError, ModelLoadError
from prime.encoder.pooling import pool_embeddings

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class CacheInfo(NamedTuple):
    """Cache statistics for encoding operations."""

    hits: int
    misses: int
    maxsize: int
    currsize: int


class YEncoder:
    """Y-Encoder for target content embeddings.

    Encodes text content into fixed-dimensional L2-normalized embeddings
    suitable for memory storage in MCS and as prediction targets for
    the Embedding Predictor.

    Uses HuggingFace transformers with configurable pooling strategies
    and automatic device resolution.

    Attributes:
        embedding_dim: Output embedding dimension.
        max_length: Maximum input sequence length in tokens.
        model_name: HuggingFace model identifier.

    Example:
        >>> from prime.encoder import YEncoder, YEncoderConfig
        >>> config = YEncoderConfig(model_name="all-MiniLM-L6-v2")
        >>> encoder = YEncoder(config)
        >>> embedding = encoder.encode("Hello, world!")
        >>> assert embedding.shape == (384,)
        >>> assert np.isclose(np.linalg.norm(embedding), 1.0)
    """

    def __init__(self, config: YEncoderConfig | None = None) -> None:
        """Initialize Y-Encoder with configuration.

        Args:
            config: Encoder configuration. Uses defaults if None.

        Raises:
            ModelLoadError: If model or tokenizer fails to load.
        """
        self._config = config or YEncoderConfig()
        self._device = self._resolve_device(self._config.device)

        try:
            self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self._config.model_name,
                trust_remote_code=self._config.trust_remote_code,
            )
            self._model: PreTrainedModel = AutoModel.from_pretrained(
                self._config.model_name,
                trust_remote_code=self._config.trust_remote_code,
            )
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model '{self._config.model_name}': {e}"
            ) from e

        # Validate actual model dimension matches config
        actual_dim = self._model.config.hidden_size
        if actual_dim != self._config.embedding_dim:
            raise ModelLoadError(
                f"Model dimension mismatch: config specifies {self._config.embedding_dim}, "
                f"but model '{self._config.model_name}' produces {actual_dim}. "
                f"Update config.embedding_dim to {actual_dim}."
            )

        # Set up optional LRU cache
        self._cache_enabled = self._config.cache_size > 0
        self._cached_encode = self._setup_cache() if self._cache_enabled else None

    def _setup_cache(self) -> Any:
        """Set up LRU cache for encoding.

        Returns:
            A cached version of _encode_uncached bound to this instance.
        """
        @lru_cache(maxsize=self._config.cache_size)
        def cached_encode(cache_key: str) -> tuple[float, ...]:
            # Cache stores tuples (hashable), convert back to array when retrieved
            embedding = self._encode_uncached(cache_key)
            return tuple(embedding.tolist())

        return cached_encode

    @staticmethod
    def _compute_cache_key(text: str) -> str:
        """Compute cache key for text using SHA256.

        Args:
            text: Input text.

        Returns:
            SHA256 hexdigest of text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device specification to torch.device.

        Auto-detection follows: cuda → mps → cpu fallback chain.

        Args:
            device: Device specification ('auto', 'cuda', 'mps', 'cpu').

        Returns:
            Resolved torch.device.
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self._config.embedding_dim

    @property
    def max_length(self) -> int:
        """Return maximum input sequence length."""
        return self._config.max_length

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._config.model_name

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode single text to embedding vector.

        Args:
            text: Input text to encode. Must be non-empty.

        Returns:
            1D numpy array of shape (embedding_dim,) with float32 dtype.
            Output is L2-normalized if configured.

        Raises:
            EncodingError: If text is empty or whitespace-only.
        """
        self._validate_input(text)

        if self._cache_enabled and self._cached_encode is not None:
            cache_key = self._compute_cache_key(text)
            cached_tuple = self._cached_encode(cache_key)
            return np.array(cached_tuple, dtype=np.float32)

        return self._encode_uncached(text)

    def _encode_uncached(self, text: str) -> NDArray[np.float32]:
        """Encode text without caching.

        Args:
            text: Input text to encode.

        Returns:
            1D numpy array of shape (embedding_dim,) with float32 dtype.
        """
        with torch.no_grad():
            # Tokenize with truncation
            inputs = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Forward pass
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Pool embeddings
            pooled = pool_embeddings(
                hidden_states,
                inputs["attention_mask"],
                self._config.pooling_mode,
            )

            # L2 normalize if configured
            if self._config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            # Convert to numpy
            embedding = pooled.squeeze(0).cpu().numpy().astype(np.float32)

        return embedding

    def encode_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Encode batch of texts to embedding vectors.

        Args:
            texts: List of input texts. Each must be non-empty.

        Returns:
            List of 1D numpy arrays, each of shape (embedding_dim,).

        Raises:
            EncodingError: If any text is empty or whitespace-only.
        """
        if not texts:
            return []

        for i, text in enumerate(texts):
            try:
                self._validate_input(text)
            except EncodingError as e:
                raise EncodingError(f"Error in text at index {i}: {e}") from e

        with torch.no_grad():
            # Tokenize batch with padding and truncation
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Forward pass
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Pool embeddings
            pooled = pool_embeddings(
                hidden_states,
                inputs["attention_mask"],
                self._config.pooling_mode,
            )

            # L2 normalize if configured
            if self._config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            # Convert to list of numpy arrays
            embeddings = [
                row.cpu().numpy().astype(np.float32) for row in pooled
            ]

        return embeddings

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata.

        Returns:
            Dictionary containing model configuration details.
        """
        info: dict[str, Any] = {
            "model_name": self._config.model_name,
            "embedding_dim": self._config.embedding_dim,
            "max_length": self._config.max_length,
            "pooling_mode": self._config.pooling_mode,
            "normalize": self._config.normalize,
            "device": str(self._device),
            "trust_remote_code": self._config.trust_remote_code,
            "cache_enabled": self._cache_enabled,
        }
        if self._cache_enabled and self._cached_encode is not None:
            info["cache_info"] = self.cache_info()
        return info

    def cache_info(self) -> CacheInfo | None:
        """Return cache statistics.

        Returns:
            CacheInfo with hits, misses, maxsize, currsize.
            None if caching is disabled.
        """
        if not self._cache_enabled or self._cached_encode is None:
            return None

        lru_info = self._cached_encode.cache_info()
        return CacheInfo(
            hits=lru_info.hits,
            misses=lru_info.misses,
            maxsize=lru_info.maxsize or 0,
            currsize=lru_info.currsize,
        )

    def clear_cache(self) -> None:
        """Clear the encoding cache."""
        if self._cache_enabled and self._cached_encode is not None:
            self._cached_encode.cache_clear()

    @staticmethod
    def _validate_input(text: str) -> None:
        """Validate input text.

        Args:
            text: Input text to validate.

        Raises:
            EncodingError: If text is empty or whitespace-only.
        """
        if not text:
            raise EncodingError("Input text cannot be empty.")
        if not text.strip():
            raise EncodingError("Input text cannot be whitespace-only.")
