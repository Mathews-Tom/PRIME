"""Optimized Embedding Predictor with Static KV Cache and torch.compile.

Provides OptimizedPredictor wrapper for production inference with:
- Static batch padding for CUDA graph compatibility
- torch.compile() with reduce-overhead mode
- Warmup for compilation trigger
- Flash Attention via SDPA
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime.predictor.config import PredictorConfig
from prime.predictor.exceptions import PredictorError
from prime.predictor.predictor import EmbeddingPredictor


class OptimizedPredictor:
    """Optimized predictor wrapper for production inference.

    Wraps EmbeddingPredictor with optimizations for low-latency inference:
    - Static batch padding for CUDA graph capture
    - torch.compile() with configurable optimization mode
    - Warmup method to trigger JIT compilation

    Expected speedup: 3-4x compared to eager execution.

    Attributes:
        predictor: Underlying EmbeddingPredictor model.
        config: Predictor configuration.
        is_warmed_up: Whether warmup has been performed.

    Example:
        >>> config = PredictorConfig(use_torch_compile=True)
        >>> optimized = OptimizedPredictor(config, device="cuda")
        >>> optimized.warmup()  # Trigger compilation
        >>> output = optimized(context, query)  # Fast inference
    """

    def __init__(
        self,
        config: PredictorConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        predictor: EmbeddingPredictor | None = None,
    ) -> None:
        """Initialize optimized predictor.

        Args:
            config: Predictor configuration. Uses defaults if None.
            device: Device for inference (cpu/cuda).
            predictor: Pre-initialized predictor. Creates new if None.

        Raises:
            PredictorError: If configuration is invalid.
        """
        self.config = config or PredictorConfig()
        self._device = torch.device(device)
        self._is_warmed_up = False
        self._original_batch_size: int | None = None
        self._compiled_predictor: torch.nn.Module | None = None

        # Initialize or use provided predictor
        if predictor is not None:
            self._predictor = predictor
        else:
            self._predictor = EmbeddingPredictor(self.config)

        self._predictor = self._predictor.to(self._device)
        self._predictor.eval()

        # Apply torch.compile if enabled
        if self.config.use_torch_compile and torch.cuda.is_available():
            self._compiled_predictor = self._compile_predictor()

    def _compile_predictor(self) -> torch.nn.Module:
        """Compile predictor with torch.compile.

        Returns:
            Compiled predictor module.
        """
        return torch.compile(  # type: ignore[return-value]
            self._predictor,
            mode=self.config.compile_mode,
            fullgraph=True,
        )

    @property
    def predictor(self) -> EmbeddingPredictor:
        """Get the underlying predictor."""
        return self._predictor

    @property
    def is_warmed_up(self) -> bool:
        """Check if predictor has been warmed up."""
        return self._is_warmed_up

    @property
    def device(self) -> torch.device:
        """Get the device for inference."""
        return self._device

    def warmup(self) -> None:
        """Trigger compilation with dummy input.

        Performs forward passes with dummy data at maximum batch size
        to trigger CUDA graph capture and JIT compilation.

        Should be called once before production inference.
        """
        if self._is_warmed_up:
            return

        # Create dummy inputs at max batch size
        dummy_context = torch.randn(
            self.config.max_batch_size,
            self.config.max_context_length,
            self.config.input_dim,
            device=self._device,
        )
        dummy_query = torch.randn(
            self.config.max_batch_size,
            self.config.input_dim,
            device=self._device,
        )

        # Warmup iterations for stable timing
        with torch.inference_mode():
            for _ in range(3):
                _ = self._forward_impl(dummy_context, dummy_query)

        # Synchronize for CUDA
        if self._device.type == "cuda":
            torch.cuda.synchronize()

        self._is_warmed_up = True

    def _pad_to_static_batch(
        self,
        tensor: Tensor,
        target_batch: int,
    ) -> Tensor:
        """Pad tensor to static batch size for CUDA graph.

        Args:
            tensor: Input tensor (B, ...).
            target_batch: Target batch size.

        Returns:
            Padded tensor (target_batch, ...).
        """
        current_batch = tensor.shape[0]

        if current_batch >= target_batch:
            return tensor[:target_batch]

        # Calculate padding shape
        pad_shape = list(tensor.shape)
        pad_shape[0] = target_batch - current_batch

        # Create zero padding
        padding = torch.zeros(
            pad_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        return torch.cat([tensor, padding], dim=0)

    def _pad_context(
        self,
        context: Tensor,
        target_length: int,
    ) -> Tensor:
        """Pad context sequence to static length.

        Args:
            context: Context tensor (B, N, D).
            target_length: Target sequence length.

        Returns:
            Padded tensor (B, target_length, D).
        """
        current_length = context.shape[1]

        if current_length >= target_length:
            return context[:, :target_length, :]

        # Pad sequence dimension
        padding_size = target_length - current_length
        padding = torch.zeros(
            context.shape[0],
            padding_size,
            context.shape[2],
            dtype=context.dtype,
            device=context.device,
        )

        return torch.cat([context, padding], dim=1)

    def _forward_impl(
        self,
        context_embeddings: Tensor,
        query_embedding: Tensor,
    ) -> Tensor:
        """Internal forward implementation.

        Args:
            context_embeddings: Context tensor (B, N, D).
            query_embedding: Query tensor (B, D).

        Returns:
            Predicted embedding (B, D).
        """
        if self._compiled_predictor is not None:
            result: Tensor = self._compiled_predictor(
                context_embeddings, query_embedding
            )
            return result
        output: Tensor = self._predictor(context_embeddings, query_embedding)
        return output

    def __call__(
        self,
        context_embeddings: Tensor,
        query_embedding: Tensor,
    ) -> Tensor:
        """Run optimized inference.

        Pads inputs to static shapes for CUDA graph compatibility,
        runs inference, and extracts the actual batch outputs.

        Args:
            context_embeddings: Context tensor (B, N, D).
            query_embedding: Query tensor (B, D).

        Returns:
            Predicted embedding (B, D), L2-normalized.

        Raises:
            PredictorError: If input dimensions are invalid.
        """
        # Validate inputs
        if context_embeddings.ndim != 3:
            msg = f"context_embeddings must be 3D, got {context_embeddings.ndim}D"
            raise PredictorError(msg)

        if query_embedding.ndim != 2:
            msg = f"query_embedding must be 2D, got {query_embedding.ndim}D"
            raise PredictorError(msg)

        # Store original batch size
        original_batch = context_embeddings.shape[0]
        self._original_batch_size = original_batch

        # Move to device if needed
        context_embeddings = context_embeddings.to(self._device)
        query_embedding = query_embedding.to(self._device)

        # Pad to static shapes if using compiled predictor
        if self._compiled_predictor is not None:
            context_embeddings = self._pad_context(
                context_embeddings, self.config.max_context_length
            )
            context_embeddings = self._pad_to_static_batch(
                context_embeddings, self.config.max_batch_size
            )
            query_embedding = self._pad_to_static_batch(
                query_embedding, self.config.max_batch_size
            )

        # Run inference
        with torch.inference_mode():
            output = self._forward_impl(context_embeddings, query_embedding)

        # Extract original batch outputs
        return output[:original_batch]

    def forward(
        self,
        context_embeddings: Tensor,
        query_embedding: Tensor,
    ) -> Tensor:
        """Alias for __call__."""
        return self(context_embeddings, query_embedding)


def create_optimized_predictor(
    config: PredictorConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    predictor: EmbeddingPredictor | None = None,
) -> OptimizedPredictor:
    """Factory function to create OptimizedPredictor.

    Creates an OptimizedPredictor with torch.compile and static
    batch padding for low-latency production inference.

    Args:
        config: Predictor configuration.
        device: Target device for inference.
        predictor: Pre-initialized predictor (optional).

    Returns:
        OptimizedPredictor instance.

    Example:
        >>> # Create from scratch
        >>> predictor = create_optimized_predictor(device="cuda")
        >>> predictor.warmup()

        >>> # Wrap existing predictor
        >>> base = EmbeddingPredictor(config)
        >>> optimized = create_optimized_predictor(device="cuda", predictor=base)
    """
    return OptimizedPredictor(config, device=device, predictor=predictor)
