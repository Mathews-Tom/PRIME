"""PyTorch Lightning training module for Embedding Predictor.

Provides PredictorLightningModule for training with InfoNCE loss,
Y-Encoder targets, and configurable optimizers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from prime.predictor import EmbeddingPredictor, TrainingConfig
from prime.training.losses import info_nce_loss

if TYPE_CHECKING:
    import lightning as L  # type: ignore[import-not-found]

    from prime.encoder import YEncoder


class PredictorLightningModule:
    """Lightning module for training EmbeddingPredictor.

    Trains the predictor to predict Y-Encoder embeddings of target
    content using InfoNCE contrastive loss with in-batch negatives.

    Note: This class is designed to work with PyTorch Lightning but
    does not inherit from LightningModule directly to avoid import
    issues when lightning is not installed. Use as_lightning_module()
    to get a proper Lightning module.

    Attributes:
        predictor: The EmbeddingPredictor model.
        y_encoder: Y-Encoder for encoding targets.
        config: Training configuration.
    """

    def __init__(
        self,
        predictor: EmbeddingPredictor,
        y_encoder: YEncoder,
        config: TrainingConfig | None = None,
    ) -> None:
        """Initialize the training module.

        Args:
            predictor: EmbeddingPredictor model to train.
            y_encoder: Y-Encoder for encoding target content.
            config: Training configuration. Uses defaults if None.
        """
        self.predictor = predictor
        self.y_encoder = y_encoder
        self.config = config or TrainingConfig()

    def compute_loss(
        self,
        predicted: torch.Tensor,
        target_texts: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute InfoNCE loss and metrics.

        Args:
            predicted: Predicted embeddings (B, D).
            target_texts: Target content strings for Y-Encoder.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        # Encode targets with Y-Encoder (returns list of numpy arrays)
        with torch.no_grad():
            target_arrays = self.y_encoder.encode_batch(target_texts)

        # Convert to tensor and stack
        positive = torch.from_numpy(np.stack(target_arrays, axis=0))
        positive = positive.to(predicted.device, dtype=predicted.dtype)

        # Compute InfoNCE loss
        loss = info_nce_loss(
            predicted,
            positive,
            temperature=self.config.temperature,
        )

        # Compute metrics
        with torch.no_grad():
            # Accuracy: is the positive the most similar?
            pred_norm = F.normalize(predicted, dim=-1)
            pos_norm = F.normalize(positive, dim=-1)
            similarity = torch.mm(pred_norm, pos_norm.t())
            pred_labels = similarity.argmax(dim=-1)
            true_labels = torch.arange(
                len(pred_labels),
                device=pred_labels.device,
            )
            accuracy = (pred_labels == true_labels).float().mean()

            # Mean cosine similarity with positive
            pos_similarity = (pred_norm * pos_norm).sum(dim=-1).mean()

        metrics = {
            "accuracy": accuracy.item(),
            "pos_similarity": pos_similarity.item(),
        }

        return loss, metrics

    def training_step(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute single training step.

        Args:
            batch: Dict containing:
                - context_embeddings: (B, N, D) tensor
                - query_embedding: (B, D) tensor
                - target_content: List of B strings

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        context = batch["context_embeddings"]
        query = batch["query_embedding"]
        target_content = batch["target_content"]

        # Forward pass
        predicted = self.predictor(context, query)

        # Compute loss and metrics
        return self.compute_loss(predicted, target_content)

    def validation_step(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute single validation step.

        Args:
            batch: Same format as training_step.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        self.predictor.eval()
        with torch.no_grad():
            context = batch["context_embeddings"]
            query = batch["query_embedding"]
            target_content = batch["target_content"]

            # Forward pass
            predicted = self.predictor(context, query)

            # Compute loss and metrics
            loss, metrics = self.compute_loss(predicted, target_content)

        return loss, metrics

    def configure_optimizers(self) -> tuple[AdamW, CosineAnnealingWarmRestarts]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Tuple of (optimizer, scheduler).
        """
        optimizer = AdamW(
            self.predictor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.warmup_steps,
            T_mult=2,
        )

        return optimizer, scheduler


def create_lightning_module(
    predictor: EmbeddingPredictor,
    y_encoder: YEncoder,
    config: TrainingConfig | None = None,
) -> L.LightningModule:
    """Create a PyTorch Lightning module for training.

    This function requires PyTorch Lightning to be installed.

    Args:
        predictor: EmbeddingPredictor model to train.
        y_encoder: Y-Encoder for encoding target content.
        config: Training configuration. Uses defaults if None.

    Returns:
        A LightningModule ready for training.

    Raises:
        ImportError: If PyTorch Lightning is not installed.
    """
    try:
        import lightning as L
    except ImportError as e:
        msg = "PyTorch Lightning required. Install with: pip install lightning"
        raise ImportError(msg) from e

    training_config = config or TrainingConfig()
    wrapper = PredictorLightningModule(predictor, y_encoder, training_config)

    class _LightningWrapper(L.LightningModule):  # type: ignore[misc]
        """Lightning wrapper for PredictorLightningModule."""

        def __init__(self) -> None:
            super().__init__()
            self.wrapper = wrapper
            self.save_hyperparameters(ignore=["predictor", "y_encoder"])

        def forward(
            self,
            context_embeddings: torch.Tensor,
            query_embedding: torch.Tensor,
        ) -> torch.Tensor:
            return self.wrapper.predictor(  # type: ignore[no-any-return]
                context_embeddings, query_embedding
            )

        def training_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
        ) -> torch.Tensor:
            loss, metrics = self.wrapper.training_step(batch)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/accuracy", metrics["accuracy"], sync_dist=True)
            self.log(
                "train/pos_similarity",
                metrics["pos_similarity"],
                sync_dist=True,
            )
            return loss

        def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
        ) -> torch.Tensor:
            loss, metrics = self.wrapper.validation_step(batch)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            self.log(
                "val/accuracy",
                metrics["accuracy"],
                prog_bar=True,
                sync_dist=True,
            )
            self.log("val/pos_similarity", metrics["pos_similarity"], sync_dist=True)
            return loss

        def configure_optimizers(self) -> dict[str, Any]:
            optimizer, scheduler = self.wrapper.configure_optimizers()
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        def on_before_optimizer_step(
            self,
            optimizer: Any,  # noqa: ARG002
        ) -> None:
            if self.wrapper.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.wrapper.predictor.parameters(),
                    self.wrapper.config.gradient_clip_val,
                )

    return _LightningWrapper()
