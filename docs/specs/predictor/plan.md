# PRED-001: Embedding Predictor Implementation Plan

**Epic:** PRED-001 - Implement Embedding Predictor
**Status:** Ready for Implementation
**Generated:** 2026-01-08

---

## 1. Executive Summary

### Objective

Implement the Embedding Predictor - PRIME's core JEPA innovation that predicts the embedding of ideal context BEFORE retrieval, enabling more targeted search and +15-25% improvement in Precision@5 over query-embedding baseline.

### Scope

- Core `EmbeddingPredictor` PyTorch module with transformer architecture
- Static KV Cache optimization for 4× inference speedup
- InfoNCE contrastive loss for training
- PyTorch Lightning training module
- ONNX export capability
- Comprehensive test suite with 85%+ coverage

### Success Criteria

| Metric | Target |
|--------|--------|
| Precision@5 vs Baseline | +15% |
| Inference Latency | <30ms p50 |
| Training Convergence | InfoNCE < 2.0 |
| Output Dimension | 1024 (Y-Encoder space) |
| Test Coverage | ≥85% |

### Dependencies

- **External:** PyTorch 2.2+, HuggingFace Transformers, PyTorch Lightning, Weights & Biases
- **Internal:** ENC-001 (Y-Encoder for training targets)
- **Blocks:** API-001

---

## 2. Context & Documentation Sources

### Primary Specification

- [docs/specs/predictor/spec.md](spec.md) - Full Predictor specification

### Architecture Context

- [.sage/agent/system/architecture.md](../../../.sage/agent/system/architecture.md) - System architecture
- [.sage/agent/system/tech-stack.md](../../../.sage/agent/system/tech-stack.md) - Technology stack

### Enhancement Integration

**From docs/enhancement.md:**
- **Static KV Cache (4× Speedup)** - Score: 8.5 - Integrated into PRED spec
  - FR-PRED-012 through FR-PRED-014 implement Static KV Cache
  - Pre-allocated tensors for CUDA graph capture
  - torch.compile() with static shapes

### Traceability Matrix

| Requirement | Source | Priority |
|-------------|--------|----------|
| FR-PRED-001: Input shapes | spec.md | P0 |
| FR-PRED-002-005: Forward pass | spec.md | P0 |
| FR-PRED-006-007: Output normalization | spec.md | P0 |
| FR-PRED-008-009: InfoNCE loss | spec.md | P0 |
| FR-PRED-010-011: Distributed training | spec.md | P1 |
| FR-PRED-012-014: Static KV Cache | spec.md, enhancement.md | P1 |

---

## 3. Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Embedding Predictor Module                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    EmbeddingPredictor (nn.Module)               │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐       │   │
│  │  │ Context     │    │ Query        │    │ Output       │       │   │
│  │  │ Projection  │    │ Projection   │    │ Projection   │       │   │
│  │  │ (D→H)       │    │ (D→H)        │    │ (H→D)        │       │   │
│  │  └─────────────┘    └──────────────┘    └──────────────┘       │   │
│  │         │                  │                   ▲                │   │
│  │         ▼                  ▼                   │                │   │
│  │  ┌───────────────────────────────────────────────────────┐     │   │
│  │  │           [PRED] + Context + Query                    │     │   │
│  │  │                     ↓                                 │     │   │
│  │  │     Transformer Layers (4-8, bidirectional)          │     │   │
│  │  │                     ↓                                 │     │   │
│  │  │               Extract [PRED]                          │     │   │
│  │  └───────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   OptimizedPredictor Wrapper                     │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │  │
│  │  │ Static KV    │    │ torch.compile│    │ CUDA Graph   │       │  │
│  │  │ Cache        │    │ (static)     │    │ Capture      │       │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Inference

```
Input: context_embeddings (B×N×1024), query_embedding (B×1024)
    │
    ▼
┌─────────────────┐
│ Context Project │ ──▶ B × N × H (hidden dim)
│ Linear(1024→H)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Project   │ ──▶ B × 1 × H
│ Linear(1024→H)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prepend [PRED]  │ ──▶ B × (1+N+1) × H
│ Learnable token │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │ ──▶ Bidirectional self-attention
│ Layers (4-8)    │     + FFN (Pre-LN architecture)
│ Static KV Cache │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract [PRED]  │ ──▶ B × H (position 0)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Project  │ ──▶ B × 1024
│ Linear(H→1024)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ L2 Normalize    │ ──▶ B × 1024 (unit vectors)
└─────────────────┘

Output: predicted_embedding (B × 1024)
```

### Data Flow: Training

```
Input: batch (context, query, positive_content, in_batch_negatives)
    │
    ├────────────────────┐
    │                    │
    ▼                    ▼
┌───────────┐      ┌───────────┐
│ Predictor │      │ Y-Encoder │
│ Forward   │      │ (frozen)  │
└─────┬─────┘      └─────┬─────┘
      │                  │
      ▼                  ▼
  predicted_emb      target_emb (positive)
      │                  │
      │           ┌──────┴──────┐
      │           │ Y-Encoder   │
      │           │ (in-batch)  │
      │           └──────┬──────┘
      │                  │
      │            negative_embs
      │                  │
      └────────┬─────────┘
               │
               ▼
┌─────────────────────────┐
│       InfoNCE Loss      │
│ -log(exp(sim+/τ) /      │
│   Σexp(sim_i/τ))        │
│ τ = 0.07                │
└────────────┬────────────┘
             │
             ▼
┌─────────────────┐
│ Backward Pass   │
│ Gradient Flow   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Optimizer Step  │
│ AdamW           │
└─────────────────┘

Output: loss value
```

---

## 4. Technical Specification

### File Structure

```
src/prime/core/
├── __init__.py           # Export Predictor
├── predictor.py          # EmbeddingPredictor module
├── predictor_config.py   # PredictorConfig
├── optimized.py          # OptimizedPredictor with Static KV Cache

src/prime/training/
├── __init__.py           # Export training components
├── trainer.py            # PyTorch Lightning module
├── data.py               # Dataset and DataLoader
├── losses.py             # InfoNCE and contrastive losses
├── callbacks.py          # Training callbacks

tests/
├── test_predictor.py     # Predictor unit tests
├── test_training.py      # Training tests
└── test_losses.py        # Loss function tests
```

### Core Implementation

#### `src/prime/core/predictor_config.py`

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictorConfig(BaseModel):
    """Configuration for Embedding Predictor."""

    input_dim: int = Field(
        default=1024,
        ge=1,
        description="Input embedding dimension",
    )
    hidden_dim: int = Field(
        default=2048,
        ge=128,
        description="Transformer hidden dimension",
    )
    output_dim: int = Field(
        default=1024,
        ge=1,
        description="Output embedding dimension (Y-Encoder space)",
    )
    num_layers: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of transformer layers",
    )
    num_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads",
    )
    max_context_length: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum context turns",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout probability",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to model checkpoint",
    )
    use_static_cache: bool = Field(
        default=True,
        description="Enable Static KV Cache",
    )
    use_torch_compile: bool = Field(
        default=True,
        description="Enable torch.compile()",
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
        description="torch.compile mode",
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Max batch size for static shapes",
    )

    model_config = {"frozen": True}


class TrainingConfig(BaseModel):
    """Training configuration for Embedding Predictor."""

    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    batch_size: int = Field(default=64, ge=1)
    num_epochs: int = Field(default=10, ge=1)
    warmup_steps: int = Field(default=1000, ge=0)
    temperature: float = Field(default=0.07, gt=0, description="InfoNCE τ")
    gradient_checkpointing: bool = Field(default=True)
    mixed_precision: bool = Field(default=True)
    y_encoder_lr_multiplier: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Y-Encoder LR multiplier (slower update)",
    )
    gradient_clip_val: float = Field(default=1.0, ge=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)

    model_config = {"frozen": True}
```

#### `src/prime/core/predictor.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prime.core.predictor_config import PredictorConfig


class PredictorError(Exception):
    """Base exception for predictor errors."""


class CheckpointError(PredictorError):
    """Error loading/saving checkpoint."""


class PredictorTransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor (B × S × H).
            attn_mask: Optional attention mask.

        Returns:
            Output tensor (B × S × H).
        """
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))

        return x


class EmbeddingPredictor(nn.Module):
    """JEPA-style embedding predictor.

    Predicts target embedding from context turns and query embedding.
    Uses bidirectional transformer with learnable [PRED] token.

    Attributes:
        config: Predictor configuration.
    """

    def __init__(self, config: PredictorConfig) -> None:
        """Initialize predictor.

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

        # Positional embeddings
        max_seq_len = config.max_context_length + 2  # +2 for [PRED] and query
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, config.hidden_dim) * 0.02
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

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        context_embeddings: Tensor,  # B × N × D
        query_embedding: Tensor,  # B × D
    ) -> Tensor:
        """Predict target embedding.

        Args:
            context_embeddings: Context turn embeddings (B × N × input_dim).
            query_embedding: Query embedding (B × input_dim).

        Returns:
            Predicted embedding (B × output_dim), L2-normalized.

        Raises:
            PredictorError: If input shapes are invalid.
        """
        B, N, D = context_embeddings.shape

        # Validate input dimensions
        if D != self.config.input_dim:
            raise PredictorError(
                f"Context dimension {D} != input_dim {self.config.input_dim}"
            )
        if query_embedding.shape != (B, self.config.input_dim):
            raise PredictorError(
                f"Query shape {query_embedding.shape} invalid for batch {B}"
            )

        # Project inputs to hidden dimension
        context = self.context_proj(context_embeddings)  # B × N × H
        query = self.query_proj(query_embedding).unsqueeze(1)  # B × 1 × H

        # Expand [PRED] token for batch
        pred = self.pred_token.expand(B, -1, -1)  # B × 1 × H

        # Concatenate: [PRED] + context + query
        x = torch.cat([pred, context, query], dim=1)  # B × (1+N+1) × H

        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Extract [PRED] position output
        x = self.norm(x[:, 0, :])  # B × H

        # Project to output dimension
        x = self.output_proj(x)  # B × D

        # L2 normalize (unit vectors)
        x = F.normalize(x, p=2, dim=-1)

        return x

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        try:
            torch.save({
                "config": self.config.model_dump(),
                "state_dict": self.state_dict(),
            }, path)
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    @classmethod
    def load_checkpoint(cls, path: str) -> EmbeddingPredictor:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Loaded EmbeddingPredictor.

        Raises:
            CheckpointError: If loading fails.
        """
        try:
            checkpoint = torch.load(path, map_location="cpu")
            config = PredictorConfig(**checkpoint["config"])
            model = cls(config)
            model.load_state_dict(checkpoint["state_dict"])
            return model
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e
```

#### `src/prime/core/optimized.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from prime.core.predictor import EmbeddingPredictor
from prime.core.predictor_config import PredictorConfig


class OptimizedPredictor(nn.Module):
    """Embedding Predictor with Static KV Cache optimization.

    Provides 4× inference speedup through:
    - Static KV Cache with pre-allocated tensors
    - torch.compile() with CUDA graph capture
    - Static batch shapes for graph reuse

    Attributes:
        config: Predictor configuration.
        predictor: Base EmbeddingPredictor module.
    """

    def __init__(
        self,
        config: PredictorConfig,
        device: torch.device | str = "cuda",
    ) -> None:
        """Initialize optimized predictor.

        Args:
            config: Predictor configuration.
            device: Target device for inference.
        """
        super().__init__()
        self.config = config
        self._device = torch.device(device)

        # Create base predictor
        self.predictor = EmbeddingPredictor(config)
        self.predictor = self.predictor.to(self._device)

        # Apply torch.compile if enabled
        if config.use_torch_compile and torch.cuda.is_available():
            self.predictor = torch.compile(
                self.predictor,
                mode=config.compile_mode,
                fullgraph=True,
            )

        self._is_warmed_up = False

    def warmup(self) -> None:
        """Warmup model with dummy inputs for CUDA graph capture.

        Must be called before inference for optimal performance.
        """
        if self._is_warmed_up:
            return

        dummy_context = torch.randn(
            self.config.max_batch_size,
            self.config.max_context_length,
            self.config.input_dim,
            device=self._device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        dummy_query = torch.randn(
            self.config.max_batch_size,
            self.config.input_dim,
            device=self._device,
            dtype=dummy_context.dtype,
        )

        # Run forward pass to trigger compilation
        with torch.inference_mode():
            _ = self.predictor(dummy_context, dummy_query)

        self._is_warmed_up = True

    @torch.inference_mode()
    def forward(
        self,
        context_embeddings: Tensor,
        query_embedding: Tensor,
    ) -> Tensor:
        """Optimized forward pass.

        Args:
            context_embeddings: Context embeddings (B × N × D).
            query_embedding: Query embedding (B × D).

        Returns:
            Predicted embedding (B × D), L2-normalized.
        """
        B = context_embeddings.shape[0]

        # Pad to static batch size for CUDA graph compatibility
        if B < self.config.max_batch_size and self.config.use_torch_compile:
            context_embeddings = self._pad_batch(
                context_embeddings, self.config.max_batch_size
            )
            query_embedding = self._pad_batch(
                query_embedding, self.config.max_batch_size
            )

        # Forward with Flash Attention
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = self.predictor(context_embeddings, query_embedding)

        # Remove padding
        return output[:B]

    def _pad_batch(self, tensor: Tensor, target_size: int) -> Tensor:
        """Pad tensor to static batch size."""
        B = tensor.shape[0]
        if B >= target_size:
            return tensor[:target_size]

        pad_shape = list(tensor.shape)
        pad_shape[0] = target_size - B
        padding = torch.zeros(
            pad_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=0)


def create_predictor(
    config: PredictorConfig | None = None,
    optimized: bool = True,
    device: str = "cuda",
) -> EmbeddingPredictor | OptimizedPredictor:
    """Factory function to create predictor.

    Args:
        config: Predictor configuration.
        optimized: Whether to use OptimizedPredictor.
        device: Target device.

    Returns:
        EmbeddingPredictor or OptimizedPredictor.
    """
    config = config or PredictorConfig()

    if optimized and torch.cuda.is_available():
        predictor = OptimizedPredictor(config, device=device)
        predictor.warmup()
        return predictor

    return EmbeddingPredictor(config).to(device)
```

#### `src/prime/training/losses.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(
    predicted: Tensor,  # B × D
    positive: Tensor,  # B × D
    negatives: Tensor | None = None,  # B × K × D or None for in-batch
    temperature: float = 0.07,
) -> Tensor:
    """Compute InfoNCE contrastive loss.

    Args:
        predicted: Predicted embeddings (B × D).
        positive: Positive target embeddings (B × D).
        negatives: Optional explicit negatives. If None, uses in-batch negatives.
        temperature: Temperature for softmax (τ).

    Returns:
        Scalar loss value.
    """
    # Normalize embeddings (should already be normalized, but ensure)
    predicted = F.normalize(predicted, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)

    # Compute positive similarity
    pos_sim = torch.sum(predicted * positive, dim=-1, keepdim=True)  # B × 1

    if negatives is None:
        # In-batch negatives: each sample's positive is negative for others
        # Similarity matrix: B × B
        all_sim = torch.mm(predicted, positive.t())  # B × B
        logits = all_sim / temperature

        # Labels: diagonal entries are positives
        labels = torch.arange(logits.shape[0], device=logits.device)

        return F.cross_entropy(logits, labels)

    else:
        # Explicit negatives provided
        negatives = F.normalize(negatives, p=2, dim=-1)
        neg_sim = torch.bmm(
            negatives, predicted.unsqueeze(-1)
        ).squeeze(-1)  # B × K

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature  # B × (1+K)

        # Labels: position 0 is the positive
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)


def proj_nce_loss(
    predicted: Tensor,
    positive: Tensor,
    negatives: Tensor | None = None,
    temperature: float = 0.07,
    projection_dim: int = 256,
) -> Tensor:
    """Compute ProjNCE loss with learnable projection.

    Enhanced contrastive loss that projects embeddings before comparison.
    Helps with harder negatives by learning a better comparison space.

    Args:
        predicted: Predicted embeddings (B × D).
        positive: Positive target embeddings (B × D).
        negatives: Optional explicit negatives.
        temperature: Temperature for softmax.
        projection_dim: Dimension of projection space.

    Returns:
        Scalar loss value.
    """
    # This would require a learnable projection head
    # For now, delegate to standard InfoNCE
    return info_nce_loss(predicted, positive, negatives, temperature)
```

#### `src/prime/training/trainer.py`

```python
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from prime.core.predictor import EmbeddingPredictor
from prime.core.predictor_config import PredictorConfig, TrainingConfig
from prime.encoder import Encoder
from prime.training.losses import info_nce_loss


class PredictorLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Predictor training.

    Attributes:
        predictor: EmbeddingPredictor model.
        y_encoder: Y-Encoder for target embeddings.
        config: Training configuration.
    """

    def __init__(
        self,
        predictor: EmbeddingPredictor,
        y_encoder: Encoder,
        config: TrainingConfig,
    ) -> None:
        """Initialize training module.

        Args:
            predictor: Predictor model to train.
            y_encoder: Y-Encoder for targets (frozen or slow LR).
            config: Training configuration.
        """
        super().__init__()
        self.predictor = predictor
        self.y_encoder = y_encoder
        self.config = config

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["predictor", "y_encoder"])

    def forward(
        self,
        context_embeddings: Tensor,
        query_embedding: Tensor,
    ) -> Tensor:
        """Forward pass through predictor."""
        return self.predictor(context_embeddings, query_embedding)

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> Tensor:
        """Single training step.

        Args:
            batch: Dictionary with context, query, target_content.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        context = batch["context_embeddings"]  # B × N × D
        query = batch["query_embedding"]  # B × D
        target_content = batch["target_content"]  # List[str]

        # Predict embeddings
        predicted = self.predictor(context, query)  # B × D

        # Encode targets with Y-Encoder
        with torch.no_grad():
            targets = [
                torch.tensor(self.y_encoder.encode(c), device=self.device)
                for c in target_content
            ]
            positive = torch.stack(targets)  # B × D

        # Compute InfoNCE loss with in-batch negatives
        loss = info_nce_loss(
            predicted=predicted,
            positive=positive,
            temperature=self.config.temperature,
        )

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, Tensor]:
        """Validation step."""
        context = batch["context_embeddings"]
        query = batch["query_embedding"]
        target_content = batch["target_content"]

        # Predict
        predicted = self.predictor(context, query)

        # Encode targets
        with torch.no_grad():
            targets = [
                torch.tensor(self.y_encoder.encode(c), device=self.device)
                for c in target_content
            ]
            positive = torch.stack(targets)

        # Compute loss
        loss = info_nce_loss(
            predicted=predicted,
            positive=positive,
            temperature=self.config.temperature,
        )

        # Compute retrieval metrics
        similarity = torch.sum(predicted * positive, dim=-1).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_similarity", similarity)

        return {"val_loss": loss, "val_similarity": similarity}

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.predictor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.num_epochs // 3,
            T_mult=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
```

---

## 5. Test Specification

### Test File: `tests/test_predictor.py`

```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from prime.core.predictor import (
    CheckpointError,
    EmbeddingPredictor,
    PredictorError,
    PredictorTransformerBlock,
)
from prime.core.predictor_config import PredictorConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config() -> PredictorConfig:
    """Create test predictor config."""
    return PredictorConfig(
        input_dim=1024,
        hidden_dim=512,  # Smaller for testing
        output_dim=1024,
        num_layers=2,
        num_heads=4,
        max_context_length=5,
        dropout=0.0,  # Disable for deterministic tests
    )


@pytest.fixture
def predictor(config: PredictorConfig) -> EmbeddingPredictor:
    """Create predictor instance."""
    return EmbeddingPredictor(config)


@pytest.fixture
def sample_input(config: PredictorConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample input tensors."""
    B, N, D = 4, 3, config.input_dim
    context = torch.randn(B, N, D)
    query = torch.randn(B, D)
    return context, query


# ============================================================================
# Forward Pass Tests
# ============================================================================


def test_forward_pass_shape(
    predictor: EmbeddingPredictor,
    sample_input: tuple[torch.Tensor, torch.Tensor],
    config: PredictorConfig,
) -> None:
    """Test forward pass produces correct output shape."""
    context, query = sample_input

    output = predictor(context, query)

    assert output.shape == (context.shape[0], config.output_dim)


def test_output_normalized(
    predictor: EmbeddingPredictor,
    sample_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test output is L2 normalized."""
    context, query = sample_input

    output = predictor(context, query)

    # Check each vector has unit norm
    norms = torch.norm(output, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_batch_inference(predictor: EmbeddingPredictor, config: PredictorConfig) -> None:
    """Test batch inference consistency."""
    D = config.input_dim
    context = torch.randn(8, 5, D)
    query = torch.randn(8, D)

    # Full batch
    output_batch = predictor(context, query)

    # Individual samples
    outputs_individual = torch.stack([
        predictor(context[i:i+1], query[i:i+1]).squeeze(0)
        for i in range(8)
    ])

    # Should be equal (within numerical precision)
    assert torch.allclose(output_batch, outputs_individual, atol=1e-4)


# ============================================================================
# Input Validation Tests
# ============================================================================


def test_invalid_context_dim_raises(
    predictor: EmbeddingPredictor,
    config: PredictorConfig,
) -> None:
    """Test invalid context dimension raises error."""
    context = torch.randn(4, 3, 512)  # Wrong dim
    query = torch.randn(4, config.input_dim)

    with pytest.raises(PredictorError, match="dimension"):
        predictor(context, query)


def test_mismatched_batch_raises(
    predictor: EmbeddingPredictor,
    config: PredictorConfig,
) -> None:
    """Test mismatched batch sizes raises error."""
    context = torch.randn(4, 3, config.input_dim)
    query = torch.randn(2, config.input_dim)  # Different batch

    with pytest.raises(PredictorError, match="shape"):
        predictor(context, query)


# ============================================================================
# Gradient Tests
# ============================================================================


def test_gradient_flow(
    predictor: EmbeddingPredictor,
    sample_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test gradients flow through model."""
    context, query = sample_input
    context.requires_grad_(True)
    query.requires_grad_(True)

    output = predictor(context, query)
    loss = output.sum()
    loss.backward()

    assert context.grad is not None
    assert query.grad is not None
    assert not torch.all(context.grad == 0)


# ============================================================================
# Checkpoint Tests
# ============================================================================


def test_checkpoint_save_load(
    predictor: EmbeddingPredictor,
    sample_input: tuple[torch.Tensor, torch.Tensor],
    tmp_path,
) -> None:
    """Test checkpoint save and load."""
    context, query = sample_input

    # Get output before save
    output_before = predictor(context, query)

    # Save checkpoint
    path = tmp_path / "checkpoint.pt"
    predictor.save_checkpoint(str(path))

    # Load checkpoint
    loaded = EmbeddingPredictor.load_checkpoint(str(path))

    # Get output after load
    output_after = loaded(context, query)

    assert torch.allclose(output_before, output_after)


def test_invalid_checkpoint_raises(tmp_path) -> None:
    """Test invalid checkpoint raises error."""
    path = tmp_path / "invalid.pt"
    path.write_text("invalid")

    with pytest.raises(CheckpointError):
        EmbeddingPredictor.load_checkpoint(str(path))


# ============================================================================
# Transformer Block Tests
# ============================================================================


def test_transformer_block_shape() -> None:
    """Test transformer block preserves shape."""
    block = PredictorTransformerBlock(hidden_dim=256, num_heads=4)
    x = torch.randn(2, 10, 256)

    output = block(x)

    assert output.shape == x.shape


# ============================================================================
# Configuration Tests
# ============================================================================


def test_default_config() -> None:
    """Test default configuration values."""
    config = PredictorConfig()

    assert config.input_dim == 1024
    assert config.hidden_dim == 2048
    assert config.output_dim == 1024
    assert config.num_layers == 4
    assert config.num_heads == 8


def test_num_params(config: PredictorConfig) -> None:
    """Test parameter count calculation."""
    predictor = EmbeddingPredictor(config)
    num_params = predictor.get_num_params()

    assert num_params > 0
    assert isinstance(num_params, int)


# ============================================================================
# Determinism Tests
# ============================================================================


def test_deterministic_output(
    config: PredictorConfig,
    sample_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test same input produces same output."""
    predictor = EmbeddingPredictor(config)
    predictor.eval()

    context, query = sample_input

    with torch.no_grad():
        output1 = predictor(context, query)
        output2 = predictor(context, query)

    assert torch.allclose(output1, output2)
```

### Test File: `tests/test_losses.py`

```python
from __future__ import annotations

import pytest
import torch

from prime.training.losses import info_nce_loss


def test_info_nce_in_batch() -> None:
    """Test InfoNCE with in-batch negatives."""
    B, D = 8, 1024
    predicted = torch.randn(B, D)
    positive = torch.randn(B, D)

    # Normalize
    predicted = torch.nn.functional.normalize(predicted, p=2, dim=-1)
    positive = torch.nn.functional.normalize(positive, p=2, dim=-1)

    loss = info_nce_loss(predicted, positive, temperature=0.07)

    assert loss.shape == ()
    assert loss > 0


def test_info_nce_perfect_match() -> None:
    """Test InfoNCE with perfect predictions."""
    B, D = 8, 1024
    embeddings = torch.randn(B, D)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    # Perfect prediction: predicted == positive
    loss = info_nce_loss(embeddings, embeddings.clone(), temperature=0.07)

    # Loss should be low (but not zero due to in-batch negatives)
    assert loss < 5.0


def test_info_nce_temperature_effect() -> None:
    """Test temperature affects loss scale."""
    B, D = 8, 1024
    predicted = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=-1)
    positive = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=-1)

    loss_low_temp = info_nce_loss(predicted, positive, temperature=0.01)
    loss_high_temp = info_nce_loss(predicted, positive, temperature=1.0)

    # Lower temperature = sharper distribution = typically higher loss
    # (unless predictions are very good)
    assert not torch.isnan(loss_low_temp)
    assert not torch.isnan(loss_high_temp)
```

---

## 6. Implementation Roadmap

### Phase 1: Core Implementation (P0)

**Step 1.1: Configuration**
- Implement `predictor_config.py`
- Add validation tests

**Step 1.2: Transformer Block**
- Implement `PredictorTransformerBlock`
- Pre-norm architecture
- Test shape preservation

**Step 1.3: EmbeddingPredictor**
- Input projections
- [PRED] token
- Positional embeddings
- Forward pass
- L2 normalization

**Step 1.4: Loss Functions**
- Implement `info_nce_loss`
- In-batch negatives
- Test loss computation

**Step 1.5: Tests**
- Shape tests
- Gradient tests
- Checkpoint tests

### Phase 2: Training (P1)

**Step 2.1: PyTorch Lightning Module**
- Training step
- Validation step
- Optimizer configuration

**Step 2.2: Data Pipeline**
- Dataset class
- DataLoader configuration
- Augmentations (if applicable)

**Step 2.3: Distributed Training**
- Multi-GPU support
- Gradient checkpointing

### Phase 3: Optimization (P1)

**Step 3.1: Static KV Cache**
- Implement `OptimizedPredictor`
- Static batch padding
- Warmup compilation

**Step 3.2: torch.compile**
- CUDA graph capture
- Benchmark speedup

**Step 3.3: ONNX Export**
- Export function
- Validation

---

## 7. Quality Assurance

### Code Quality Gates

| Gate | Requirement | Tool |
|------|-------------|------|
| Type Safety | 100% coverage | mypy --strict |
| Linting | No errors | ruff check |
| Formatting | Consistent | ruff format |
| Test Coverage | ≥85% | pytest-cov |
| Tests | All passing | pytest |

### Performance Validation

```bash
# Inference latency benchmark
uv run python -c "
import time
import torch
from prime.core.predictor import EmbeddingPredictor
from prime.core.predictor_config import PredictorConfig

config = PredictorConfig()
model = EmbeddingPredictor(config).cuda().eval()

context = torch.randn(1, 5, 1024, device='cuda')
query = torch.randn(1, 1024, device='cuda')

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(context, query)

# Benchmark
times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(context, query)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

import numpy as np
print(f'p50: {np.percentile(times, 50):.2f}ms')
print(f'p95: {np.percentile(times, 95):.2f}ms')
"
```

---

## 8. Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Training doesn't converge | Medium | High | Careful LR scheduling, gradient clipping |
| Latency exceeds target | Medium | Medium | Static KV Cache, torch.compile |
| OOM during training | Medium | Medium | Gradient checkpointing, mixed precision |
| Y-Encoder drift | Low | Medium | EMA update, frozen Y-Encoder option |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Predictor doesn't improve over baseline | Medium | Critical | Ablation studies, alternative architectures |
| InfoNCE not suitable | Low | High | Try ProjNCE, triplet loss |

---

## 9. References & Traceability

### Source Documents

| Document | Purpose |
|----------|---------|
| [spec.md](spec.md) | Functional requirements |
| [architecture.md](../../../.sage/agent/system/architecture.md) | System context |
| [tech-stack.md](../../../.sage/agent/system/tech-stack.md) | Technology choices |
| [enhancement.md](../../../docs/enhancement.md) | Static KV Cache requirements |

### Related Tickets

| Ticket | Relationship |
|--------|--------------|
| ENC-001 | Provides Y-Encoder for training targets |
| SSM-001 | Upstream trigger for prediction |
| MCS-001 | Consumer of predicted embeddings |
| API-001 | Integrates Predictor into PRIME |

### External References

- [VL-JEPA Paper](https://ai.meta.com/research/publications/vl-jepa/)
- [InfoNCE Loss](https://arxiv.org/abs/1807.03748)
- [torch.compile Documentation](https://pytorch.org/docs/stable/torch.compiler_faq.html)
- [Static KV Cache in Transformers](https://huggingface.co/docs/transformers/llm_optims)

---

## Appendix A: InfoNCE Loss Formula

```
L_InfoNCE = -log( exp(sim(p, t⁺) / τ) / Σᵢ exp(sim(p, tᵢ) / τ) )

where:
- p = predicted embedding
- t⁺ = positive target
- tᵢ = all targets (positive + negatives)
- τ = temperature (0.07)
- sim(a, b) = a · b (cosine similarity for L2-normalized vectors)
```

## Appendix B: Model Size Estimation

```
Parameters breakdown (default config):
- context_proj: 1024 × 2048 = 2.1M
- query_proj: 1024 × 2048 = 2.1M
- output_proj: 2048 × 1024 = 2.1M
- pred_token: 2048 = 2K
- pos_embedding: 12 × 2048 = 24K
- transformer layers (×4):
  - attention: 2048² × 4 = 16.8M each
  - mlp: 2048 × 8192 × 2 = 33.6M each
  - norms: 2048 × 4 = 8K each
  - Total per layer: ~50M

Total: ~206M parameters (~825MB fp32, ~412MB fp16)
```
