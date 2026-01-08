"""Type definitions for Semantic State Monitor.

Defines the ActionState enum representing SSM output states and
SemanticStateUpdate for returning comprehensive update results.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActionState(str, Enum):
    """Action states emitted by SSM based on variance level.

    The SSM determines which action to recommend based on the
    smoothed variance compared against configurable thresholds.

    States:
        CONTINUE: variance < prepare_ratio * θ
            No retrieval needed, continue with existing context.

        PREPARE: prepare_ratio * θ ≤ variance < θ
            Approaching boundary, pre-warm caches for likely retrieval.

        RETRIEVE: θ ≤ variance < consolidate_ratio * θ
            Semantic boundary crossed, trigger retrieval operation.

        RETRIEVE_CONSOLIDATE: variance ≥ consolidate_ratio * θ
            Major topic shift detected, retrieve and consolidate memories.
    """

    CONTINUE = "continue"
    PREPARE = "prepare"
    RETRIEVE = "retrieve"
    RETRIEVE_CONSOLIDATE = "retrieve_consolidate"


class SemanticStateUpdate(BaseModel):
    """Result of semantic state update operation.

    Contains all information from processing a new text through the SSM,
    including variance metrics, recommended action, and the embedding
    for downstream use.

    Attributes:
        variance: Raw variance value from Ward distance calculation.
        smoothed_variance: EMA-smoothed variance for stable triggering.
        action: Recommended action based on variance thresholds.
        boundary_crossed: True if smoothed_variance >= threshold θ.
        embedding: Query embedding vector for downstream components.
        window_size: Current number of embeddings in sliding window.
        turn_number: Conversation turn count since last reset.

    Example:
        >>> result = ssm.update("How do I implement authentication?")
        >>> if result.boundary_crossed:
        ...     trigger_retrieval(result.embedding)
    """

    variance: float = Field(
        ge=0.0,
        description="Raw variance value from Ward distance calculation",
    )
    smoothed_variance: float = Field(
        ge=0.0,
        description="EMA-smoothed variance for stable triggering",
    )
    action: ActionState = Field(
        description="Recommended action based on variance thresholds",
    )
    boundary_crossed: bool = Field(
        description="True if smoothed variance exceeds threshold θ",
    )
    embedding: list[float] = Field(
        description="Query embedding vector for downstream use",
    )
    window_size: int = Field(
        ge=1,
        description="Current number of embeddings in sliding window",
    )
    turn_number: int = Field(
        ge=0,
        description="Conversation turn count since last reset",
    )

    model_config = {"frozen": True}
