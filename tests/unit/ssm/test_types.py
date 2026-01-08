"""Tests for SSM type definitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prime.ssm import ActionState, SemanticStateUpdate


class TestActionState:
    """Test ActionState enum."""

    def test_continue_value(self) -> None:
        """CONTINUE has correct string value."""
        assert ActionState.CONTINUE.value == "continue"

    def test_prepare_value(self) -> None:
        """PREPARE has correct string value."""
        assert ActionState.PREPARE.value == "prepare"

    def test_retrieve_value(self) -> None:
        """RETRIEVE has correct string value."""
        assert ActionState.RETRIEVE.value == "retrieve"

    def test_retrieve_consolidate_value(self) -> None:
        """RETRIEVE_CONSOLIDATE has correct string value."""
        assert ActionState.RETRIEVE_CONSOLIDATE.value == "retrieve_consolidate"

    def test_is_string_enum(self) -> None:
        """ActionState is a string enum."""
        assert isinstance(ActionState.CONTINUE, str)
        assert ActionState.CONTINUE == "continue"

    def test_all_values(self) -> None:
        """All expected action states exist."""
        values = {s.value for s in ActionState}
        expected = {"continue", "prepare", "retrieve", "retrieve_consolidate"}
        assert values == expected


class TestSemanticStateUpdate:
    """Test SemanticStateUpdate model."""

    def test_valid_creation(self) -> None:
        """Can create valid SemanticStateUpdate."""
        update = SemanticStateUpdate(
            variance=0.1,
            smoothed_variance=0.08,
            action=ActionState.CONTINUE,
            boundary_crossed=False,
            embedding=[0.1] * 1024,
            window_size=3,
            turn_number=5,
        )
        assert update.variance == 0.1
        assert update.smoothed_variance == 0.08
        assert update.action == ActionState.CONTINUE
        assert not update.boundary_crossed
        assert len(update.embedding) == 1024
        assert update.window_size == 3
        assert update.turn_number == 5

    def test_variance_must_be_non_negative(self) -> None:
        """variance must be >= 0."""
        with pytest.raises(ValidationError):
            SemanticStateUpdate(
                variance=-0.1,
                smoothed_variance=0.0,
                action=ActionState.CONTINUE,
                boundary_crossed=False,
                embedding=[0.1],
                window_size=1,
                turn_number=0,
            )

    def test_smoothed_variance_must_be_non_negative(self) -> None:
        """smoothed_variance must be >= 0."""
        with pytest.raises(ValidationError):
            SemanticStateUpdate(
                variance=0.0,
                smoothed_variance=-0.1,
                action=ActionState.CONTINUE,
                boundary_crossed=False,
                embedding=[0.1],
                window_size=1,
                turn_number=0,
            )

    def test_window_size_must_be_positive(self) -> None:
        """window_size must be >= 1."""
        with pytest.raises(ValidationError):
            SemanticStateUpdate(
                variance=0.0,
                smoothed_variance=0.0,
                action=ActionState.CONTINUE,
                boundary_crossed=False,
                embedding=[0.1],
                window_size=0,
                turn_number=0,
            )

    def test_turn_number_must_be_non_negative(self) -> None:
        """turn_number must be >= 0."""
        with pytest.raises(ValidationError):
            SemanticStateUpdate(
                variance=0.0,
                smoothed_variance=0.0,
                action=ActionState.CONTINUE,
                boundary_crossed=False,
                embedding=[0.1],
                window_size=1,
                turn_number=-1,
            )

    def test_immutable(self) -> None:
        """SemanticStateUpdate is frozen."""
        update = SemanticStateUpdate(
            variance=0.1,
            smoothed_variance=0.1,
            action=ActionState.CONTINUE,
            boundary_crossed=False,
            embedding=[0.1],
            window_size=1,
            turn_number=1,
        )
        with pytest.raises(ValidationError):
            update.variance = 0.5  # type: ignore[misc]

    def test_serialization(self) -> None:
        """SemanticStateUpdate can be serialized to dict."""
        update = SemanticStateUpdate(
            variance=0.1,
            smoothed_variance=0.08,
            action=ActionState.CONTINUE,
            boundary_crossed=False,
            embedding=[0.1, 0.2],
            window_size=2,
            turn_number=3,
        )
        data = update.model_dump()
        assert data["variance"] == 0.1
        assert data["action"] == "continue"
        assert data["embedding"] == [0.1, 0.2]
