"""Loss functions for PRIME training.

Provides InfoNCE contrastive loss with in-batch negatives for
training the Embedding Predictor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    predicted: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss with in-batch negatives.

    Uses cosine similarity between predicted and target embeddings.
    All other targets in the batch serve as negative examples.

    Formula:
        L = -log( exp(sim(p, t+) / tau) / sum_i exp(sim(p, t_i) / tau) )

    Args:
        predicted: Predicted embeddings (B, D), L2-normalized.
        positive: Positive target embeddings (B, D), L2-normalized.
        temperature: Softmax temperature (default 0.07).
            Lower values sharpen the distribution.

    Returns:
        Scalar loss tensor (mean over batch).

    Raises:
        ValueError: If batch sizes don't match or inputs not 2D.

    Example:
        >>> predicted = F.normalize(torch.randn(32, 1024), dim=-1)
        >>> positive = F.normalize(torch.randn(32, 1024), dim=-1)
        >>> loss = info_nce_loss(predicted, positive, temperature=0.07)
    """
    if predicted.ndim != 2 or positive.ndim != 2:
        msg = f"Inputs must be 2D: predicted {predicted.shape}, positive {positive.shape}"
        raise ValueError(msg)

    if predicted.shape[0] != positive.shape[0]:
        msg = (
            f"Batch size mismatch: predicted {predicted.shape[0]}, "
            f"positive {positive.shape[0]}"
        )
        raise ValueError(msg)

    batch_size = predicted.shape[0]

    # Ensure normalized for cosine similarity
    predicted = F.normalize(predicted, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)

    # Compute similarity matrix: B x B
    # Each row i contains similarities between predicted[i] and all positives
    similarity_matrix = torch.mm(predicted, positive.t())

    # Scale by temperature
    logits = similarity_matrix / temperature

    # Labels: diagonal entries (i, i) are the positive pairs
    labels = torch.arange(batch_size, device=logits.device)

    # Cross-entropy loss: softmax over columns, then negative log-likelihood
    loss = F.cross_entropy(logits, labels)

    return loss


def symmetric_info_nce_loss(
    predicted: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute symmetric InfoNCE loss (bidirectional).

    Averages loss from both directions:
    - predicted → positive
    - positive → predicted

    This encourages symmetric alignment between representations.

    Args:
        predicted: Predicted embeddings (B, D), L2-normalized.
        positive: Positive target embeddings (B, D), L2-normalized.
        temperature: Softmax temperature (default 0.07).

    Returns:
        Scalar loss tensor (mean of both directions).
    """
    loss_p2t = info_nce_loss(predicted, positive, temperature)
    loss_t2p = info_nce_loss(positive, predicted, temperature)

    return (loss_p2t + loss_t2p) / 2


def info_nce_loss_with_negatives(
    predicted: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE loss with explicit negatives.

    Unlike in-batch negatives, this uses provided negative examples.
    Useful when hard negatives are available.

    Args:
        predicted: Predicted embeddings (B, D).
        positive: Positive target embeddings (B, D).
        negatives: Negative embeddings (B, N, D) where N is num negatives.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    if predicted.ndim != 2 or positive.ndim != 2:
        msg = "predicted and positive must be 2D"
        raise ValueError(msg)

    if negatives.ndim != 3:
        msg = f"negatives must be 3D (B, N, D), got {negatives.shape}"
        raise ValueError(msg)

    batch_size = predicted.shape[0]

    # Normalize all
    predicted = F.normalize(predicted, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negatives = F.normalize(negatives, p=2, dim=-1)

    # Positive similarity: (B,)
    pos_sim = (predicted * positive).sum(dim=-1, keepdim=True)

    # Negative similarities: (B, N)
    neg_sim = torch.bmm(negatives, predicted.unsqueeze(-1)).squeeze(-1)

    # Concatenate: (B, 1 + N)
    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature

    # Labels: positive is at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)

    return loss
