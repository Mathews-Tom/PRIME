"""Training module for PRIME.

Provides loss functions and PyTorch Lightning training modules
for training the Embedding Predictor.

Public API:
    info_nce_loss: InfoNCE contrastive loss with in-batch negatives
    symmetric_info_nce_loss: Bidirectional InfoNCE loss
    info_nce_loss_with_negatives: InfoNCE with explicit negatives
    PredictorLightningModule: Lightning module for training

Example:
    >>> from prime.training import PredictorLightningModule, info_nce_loss
    >>> module = PredictorLightningModule(predictor, y_encoder)
"""

from __future__ import annotations

from prime.training.losses import (
    info_nce_loss,
    info_nce_loss_with_negatives,
    symmetric_info_nce_loss,
)
from prime.training.trainer import (
    PredictorLightningModule,
    create_lightning_module,
)

__all__ = [
    "PredictorLightningModule",
    "create_lightning_module",
    "info_nce_loss",
    "info_nce_loss_with_negatives",
    "symmetric_info_nce_loss",
]
