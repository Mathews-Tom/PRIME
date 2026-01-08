"""Core EmbeddingPredictor module for PRIME.

Implements JEPA-style embedding prediction using transformer architecture
with [PRED] token pattern for sequence-to-embedding prediction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prime.predictor.config import PredictorConfig
from prime.predictor.exceptions import CheckpointError, PredictorShapeError
from prime.predictor.types import CheckpointMetadata

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PredictorTransformerBlock(nn.Module):
    """Single transformer block for predictor with pre-norm architecture.

    Uses pre-norm (LayerNorm before attention/FFN) for improved training
    stability, following modern transformer best practices.

    Attributes:
        attention: Multi-head self-attention layer.
        mlp: Feed-forward network (expand → GELU → contract).
        norm1: LayerNorm before attention.
        norm2: LayerNorm before FFN.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer block.

        Args:
            hidden_dim: Hidden dimension for embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor of shape (B, S, H).

        Returns:
            Output tensor of shape (B, S, H).
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x


class EmbeddingPredictor(nn.Module):
    """JEPA-style embedding predictor for targeted retrieval.

    Predicts the embedding of ideal context BEFORE retrieval by processing
    context history and current query through a transformer with learnable
    [PRED] token. The output is a unit vector in Y-Encoder embedding space.

    Architecture:
        1. Project context and query to hidden dimension
        2. Prepend learnable [PRED] token
        3. Add positional embeddings
        4. Process through transformer layers
        5. Extract [PRED] position and project to output
        6. L2-normalize output

    Attributes:
        config: Predictor configuration.
        context_proj: Linear projection for context embeddings.
        query_proj: Linear projection for query embedding.
        output_proj: Linear projection to output dimension.
        pred_token: Learnable [PRED] token.
        pos_embed: Positional embeddings.
        layers: Stack of transformer blocks.
        final_norm: Final layer normalization.
    """

    def __init__(self, config: PredictorConfig) -> None:
        """Initialize the embedding predictor.

        Args:
            config: Predictor configuration.
        """
        super().__init__()
        self.config = config

        # Input projections
        self.context_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.query_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Learnable [PRED] token
        self.pred_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )

        # Positional embeddings: [PRED] + context + query
        max_positions = 1 + config.max_context_length + 1
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_positions, config.hidden_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            PredictorTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_shapes(
        self,
        context_embeddings: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> None:
        """Validate input tensor shapes.

        Args:
            context_embeddings: Context tensor (B, N, D).
            query_embedding: Query tensor (B, D).

        Raises:
            PredictorShapeError: If shapes are invalid.
        """
        if context_embeddings.ndim != 3:
            msg = (
                f"context_embeddings must be 3D (B, N, D), "
                f"got shape {context_embeddings.shape}"
            )
            raise PredictorShapeError(msg)

        if query_embedding.ndim != 2:
            msg = (
                f"query_embedding must be 2D (B, D), "
                f"got shape {query_embedding.shape}"
            )
            raise PredictorShapeError(msg)

        batch_size, context_len, input_dim = context_embeddings.shape

        if input_dim != self.config.input_dim:
            msg = (
                f"Context dimension {input_dim} != config.input_dim "
                f"{self.config.input_dim}"
            )
            raise PredictorShapeError(msg)

        if query_embedding.shape[0] != batch_size:
            msg = (
                f"Batch size mismatch: context has {batch_size}, "
                f"query has {query_embedding.shape[0]}"
            )
            raise PredictorShapeError(msg)

        if query_embedding.shape[1] != self.config.input_dim:
            msg = (
                f"Query dimension {query_embedding.shape[1]} != config.input_dim "
                f"{self.config.input_dim}"
            )
            raise PredictorShapeError(msg)

        if context_len > self.config.max_context_length:
            msg = (
                f"Context length {context_len} > max_context_length "
                f"{self.config.max_context_length}"
            )
            raise PredictorShapeError(msg)

        if batch_size > self.config.max_batch_size:
            msg = (
                f"Batch size {batch_size} > max_batch_size "
                f"{self.config.max_batch_size}"
            )
            raise PredictorShapeError(msg)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target embedding from context and query.

        Args:
            context_embeddings: Context turn embeddings (B, N, D).
            query_embedding: Current query embedding (B, D).

        Returns:
            Predicted target embedding (B, output_dim), L2-normalized.

        Raises:
            PredictorShapeError: If input shapes are invalid.
        """
        self._validate_shapes(context_embeddings, query_embedding)

        batch_size = context_embeddings.shape[0]

        # Project inputs to hidden dimension
        context = self.context_proj(context_embeddings)  # (B, N, H)
        query = self.query_proj(query_embedding).unsqueeze(1)  # (B, 1, H)

        # Expand [PRED] token for batch
        pred = self.pred_token.expand(batch_size, -1, -1)  # (B, 1, H)

        # Concatenate: [PRED] + context + query
        x = torch.cat([pred, context, query], dim=1)  # (B, 1+N+1, H)

        # Add positional embeddings (truncate to actual sequence length)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Apply final normalization
        x = self.final_norm(x)

        # Extract [PRED] position (first token)
        pred_output = x[:, 0, :]  # (B, H)

        # Project to output dimension
        output = self.output_proj(pred_output)  # (B, D)

        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)

        return output

    def predict(
        self,
        context_embeddings: NDArray[np.float32],
        query_embedding: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Predict embedding from numpy arrays (inference API).

        Args:
            context_embeddings: Context embeddings (N, D) or (B, N, D).
            query_embedding: Query embedding (D,) or (B, D).

        Returns:
            Predicted embedding as numpy array (D,) or (B, D).
        """
        # Handle unbatched input
        squeeze_output = False
        if context_embeddings.ndim == 2:
            context_embeddings = np.expand_dims(context_embeddings, axis=0)
            squeeze_output = True
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        # Convert to tensors
        device = next(self.parameters()).device
        context_tensor = torch.from_numpy(context_embeddings).to(device)
        query_tensor = torch.from_numpy(query_embedding).to(device)

        # Run inference
        with torch.no_grad():
            output = self.forward(context_tensor, query_tensor)

        # Convert back to numpy
        result = output.cpu().numpy()

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _get_config_hash(self) -> str:
        """Compute hash of configuration for checkpoint compatibility."""
        config_str = (
            f"{self.config.input_dim}_{self.config.hidden_dim}_"
            f"{self.config.output_dim}_{self.config.num_layers}_"
            f"{self.config.num_heads}_{self.config.max_context_length}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int = 0,
        step: int = 0,
        loss: float = 0.0,
    ) -> None:
        """Save model checkpoint with metadata.

        Args:
            path: Path to save checkpoint.
            epoch: Current training epoch.
            step: Current global step.
            loss: Current validation loss.

        Raises:
            CheckpointError: If save fails.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = CheckpointMetadata(
            epoch=epoch,
            step=step,
            loss=loss,
            config_hash=self._get_config_hash(),
        )

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "max_context_length": self.config.max_context_length,
                "dropout": self.config.dropout,
            },
            "metadata": {
                "epoch": metadata.epoch,
                "step": metadata.step,
                "loss": metadata.loss,
                "config_hash": metadata.config_hash,
            },
        }

        try:
            torch.save(checkpoint, path)
        except Exception as e:
            msg = f"Failed to save checkpoint to {path}: {e}"
            raise CheckpointError(msg) from e

    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = True,
    ) -> CheckpointMetadata:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
            strict: Whether to require exact key match.

        Returns:
            Checkpoint metadata.

        Raises:
            CheckpointError: If load fails or config mismatch.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Checkpoint not found: {path}"
            raise CheckpointError(msg)

        try:
            checkpoint: dict[str, Any] = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as e:
            msg = f"Failed to load checkpoint from {path}: {e}"
            raise CheckpointError(msg) from e

        # Verify config compatibility
        saved_hash = checkpoint.get("metadata", {}).get("config_hash", "")
        current_hash = self._get_config_hash()
        if saved_hash and saved_hash != current_hash:
            msg = (
                f"Config mismatch: checkpoint hash {saved_hash} != "
                f"current hash {current_hash}"
            )
            raise CheckpointError(msg)

        # Load state dict
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Return metadata
        meta = checkpoint.get("metadata", {})
        return CheckpointMetadata(
            epoch=meta.get("epoch", 0),
            step=meta.get("step", 0),
            loss=meta.get("loss", 0.0),
            config_hash=meta.get("config_hash", ""),
        )

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_head_dim(self) -> int:
        """Return dimension per attention head."""
        return self.config.hidden_dim // self.config.num_heads

    def export_onnx(
        self,
        path: str | Path,
        opset_version: int = 17,
    ) -> None:
        """Export model to ONNX format.

        Creates an ONNX model file for deployment in non-PyTorch environments.
        Uses dynamic axes for batch size and context length flexibility.

        Args:
            path: Output path for ONNX model file.
            opset_version: ONNX opset version (default: 17 for PyTorch 2.x).

        Raises:
            CheckpointError: If export fails.

        Example:
            >>> predictor.export_onnx("model.onnx")
            >>> # Load with ONNX Runtime
            >>> import onnxruntime as ort
            >>> session = ort.InferenceSession("model.onnx")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy inputs at typical sizes
        dummy_context = torch.randn(
            1, self.config.max_context_length, self.config.input_dim
        )
        dummy_query = torch.randn(1, self.config.input_dim)

        # Ensure model is in eval mode
        was_training = self.training
        self.eval()

        try:
            torch.onnx.export(
                self,
                (dummy_context, dummy_query),
                str(path),
                input_names=["context_embeddings", "query_embedding"],
                output_names=["predicted_embedding"],
                dynamic_axes={
                    "context_embeddings": {0: "batch", 1: "context_len"},
                    "query_embedding": {0: "batch"},
                    "predicted_embedding": {0: "batch"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
        except Exception as e:
            msg = f"Failed to export ONNX to {path}: {e}"
            raise CheckpointError(msg) from e
        finally:
            if was_training:
                self.train()


def create_predictor(config: PredictorConfig | None = None) -> EmbeddingPredictor:
    """Factory function to create EmbeddingPredictor.

    Args:
        config: Predictor configuration. Uses defaults if None.

    Returns:
        Initialized EmbeddingPredictor instance.
    """
    if config is None:
        config = PredictorConfig()
    return EmbeddingPredictor(config)
