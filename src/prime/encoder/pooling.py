"""Pooling strategies for sequence embeddings.

Implements mean, CLS, and max pooling for transformer hidden states
with proper attention mask handling.
"""

from __future__ import annotations

from typing import Literal

import torch

PoolingMode = Literal["mean", "cls", "max"]


def pool_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: PoolingMode,
) -> torch.Tensor:
    """Pool sequence hidden states into fixed-size embeddings.

    Aggregates variable-length transformer outputs into fixed-size vectors
    using attention mask weighting to ignore padding tokens.

    Args:
        hidden_states: Transformer output of shape (batch, seq_len, hidden_dim).
        attention_mask: Binary mask of shape (batch, seq_len) where 1 indicates
            valid tokens and 0 indicates padding.
        mode: Pooling strategy - 'mean', 'cls', or 'max'.

    Returns:
        Pooled embeddings of shape (batch, hidden_dim).

    Raises:
        ValueError: If mode is not one of 'mean', 'cls', 'max'.
    """
    if mode == "cls":
        return _cls_pooling(hidden_states)
    elif mode == "mean":
        return _mean_pooling(hidden_states, attention_mask)
    elif mode == "max":
        return _max_pooling(hidden_states, attention_mask)
    else:
        raise ValueError(f"Invalid pooling mode: {mode}. Must be 'mean', 'cls', or 'max'.")


def _cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract the first token (CLS) embedding.

    For BERT-style models, the [CLS] token at position 0 is trained
    to represent the entire sequence.

    Args:
        hidden_states: Shape (batch, seq_len, hidden_dim).

    Returns:
        CLS embeddings of shape (batch, hidden_dim).
    """
    return hidden_states[:, 0, :]


def _mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute attention-weighted mean of token embeddings.

    Averages token embeddings with attention mask weighting to exclude
    padding tokens. This typically produces better embeddings than CLS
    for many encoder models.

    Args:
        hidden_states: Shape (batch, seq_len, hidden_dim).
        attention_mask: Shape (batch, seq_len).

    Returns:
        Mean-pooled embeddings of shape (batch, hidden_dim).
    """
    # Expand mask to hidden dimension: (batch, seq_len) -> (batch, seq_len, hidden_dim)
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    # Sum embeddings weighted by mask
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

    # Count valid tokens per sample (avoid division by zero)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

    return sum_embeddings / sum_mask


def _max_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute element-wise max over token embeddings.

    Takes maximum value across the sequence dimension for each hidden
    dimension, masking padding tokens with -inf.

    Args:
        hidden_states: Shape (batch, seq_len, hidden_dim).
        attention_mask: Shape (batch, seq_len).

    Returns:
        Max-pooled embeddings of shape (batch, hidden_dim).
    """
    # Expand mask: (batch, seq_len) -> (batch, seq_len, hidden_dim)
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    # Set padding positions to large negative value
    hidden_states_masked = hidden_states.clone()
    hidden_states_masked[mask_expanded == 0] = float("-inf")

    # Take max over sequence dimension
    max_embeddings, _ = torch.max(hidden_states_masked, dim=1)

    return max_embeddings
