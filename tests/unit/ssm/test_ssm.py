"""Comprehensive tests for SemanticStateMonitor."""

from __future__ import annotations

import numpy as np
import pytest

from prime.ssm import ActionState, SemanticStateMonitor, SemanticStateUpdate, SSMConfig
from prime.ssm.exceptions import EncodingError, SSMError

from .conftest import MockEncoder


class TestSSMBasicFunctionality:
    """Test basic SSM operations."""

    def test_update_returns_semantic_state(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """update() returns SemanticStateUpdate with all fields."""
        result = ssm.update("Hello, how are you?")

        assert isinstance(result, SemanticStateUpdate)
        assert result.variance >= 0.0
        assert result.smoothed_variance >= 0.0
        assert isinstance(result.action, ActionState)
        assert isinstance(result.boundary_crossed, bool)
        assert len(result.embedding) == 1024
        assert result.turn_number == 1
        assert result.window_size == 1

    def test_turn_number_increments(self, ssm: SemanticStateMonitor) -> None:
        """Turn number increments with each update."""
        for i in range(5):
            result = ssm.update(f"Message {i}")
            assert result.turn_number == i + 1

    def test_window_size_grows_then_caps(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Window size grows to max then stays constant."""
        for i in range(10):
            result = ssm.update(f"Message {i}")
            expected = min(i + 1, ssm.window_size)
            assert result.window_size == expected

    def test_embedding_returned_in_result(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Update result contains the query embedding."""
        result = ssm.update("Test message")
        assert len(result.embedding) == 1024
        # Embedding should be normalized (from mock encoder)
        norm = np.linalg.norm(result.embedding)
        assert 0.99 < norm < 1.01


class TestSSMBoundaryDetection:
    """Test boundary detection logic."""

    def test_boundary_detection_on_topic_change(
        self, mock_encoder: MockEncoder
    ) -> None:
        """Boundary is detected when topic changes significantly."""
        # Create embeddings - first 4 similar, last one very different
        similar_emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        different_emb = np.array([0.0] * 512 + [1.0] + [0.0] * 511, dtype=np.float32)

        embeddings = [similar_emb.copy() for _ in range(4)]
        embeddings.append(different_emb)

        mock_encoder.set_embeddings(embeddings)

        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(
                window_size=5,
                variance_threshold=0.01,  # Low threshold for test
                smoothing_factor=0.5,
                embedding_dim=1024,
            ),
        )

        # Process same-topic messages
        for i in range(4):
            result = ssm.update(f"Topic A message {i}")
            assert not result.boundary_crossed, f"Unexpected boundary at turn {i}"

        # Process different topic - should cross boundary
        result = ssm.update("Topic B completely different")
        assert result.boundary_crossed

    def test_no_boundary_on_same_topic(self, mock_encoder: MockEncoder) -> None:
        """No boundary detected for consistent same-topic messages."""
        # Create all similar embeddings with tiny noise
        base_emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        embeddings = [base_emb + np.random.randn(1024).astype(np.float32) * 0.001 for _ in range(10)]
        for emb in embeddings:
            emb /= np.linalg.norm(emb)

        mock_encoder.set_embeddings(embeddings)

        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(
                window_size=5,
                variance_threshold=0.5,  # High threshold
                smoothing_factor=0.3,
                embedding_dim=1024,
            ),
        )

        # All messages should not cross boundary
        for i in range(10):
            result = ssm.update(f"Same topic {i}")
            assert not result.boundary_crossed

    def test_insufficient_data_returns_zero_variance(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """First update returns zero variance (can't calculate with 1 point)."""
        result = ssm.update("Single message")
        assert result.variance == 0.0
        assert result.action == ActionState.CONTINUE
        assert not result.boundary_crossed


class TestSSMEMASmoothing:
    """Test EMA smoothing behavior."""

    def test_ema_smoothing_reduces_noise(
        self, mock_encoder: MockEncoder
    ) -> None:
        """EMA smoothing reduces impact of variance spikes."""
        # Create embeddings with one spike
        base_emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        spike_emb = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)

        embeddings = [base_emb.copy() for _ in range(3)]
        embeddings.append(spike_emb)  # Spike
        embeddings.append(base_emb.copy())  # Back to normal

        mock_encoder.set_embeddings(embeddings)

        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(
                window_size=5,
                variance_threshold=0.5,
                smoothing_factor=0.3,  # Low alpha = more smoothing
                embedding_dim=1024,
            ),
        )

        variances = []
        smoothed_variances = []

        for i in range(5):
            result = ssm.update(f"Message {i}")
            variances.append(result.variance)
            smoothed_variances.append(result.smoothed_variance)

        # After spike, smoothed should be dampened compared to next raw
        # The smoothing should carry forward
        if variances[3] > 0:
            assert smoothed_variances[3] <= variances[3]

    def test_first_turn_ema_equals_raw(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """First turn smoothed variance equals raw variance."""
        result = ssm.update("First message")
        # For first turn with single point, both are 0
        assert result.smoothed_variance == result.variance

    def test_ema_formula_correctness(
        self, mock_encoder: MockEncoder
    ) -> None:
        """EMA applies correct formula: s = α*v + (1-α)*prev."""
        alpha = 0.4
        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(
                window_size=5,
                smoothing_factor=alpha,
                embedding_dim=1024,
            ),
        )

        # Get first two updates
        result1 = ssm.update("Message 1")
        prev_smoothed = result1.smoothed_variance

        result2 = ssm.update("Message 2")

        # Verify EMA formula
        expected = alpha * result2.variance + (1 - alpha) * prev_smoothed
        assert abs(result2.smoothed_variance - expected) < 1e-6


class TestSSMActionStates:
    """Test action state determination."""

    def test_continue_state_low_variance(
        self, mock_encoder: MockEncoder
    ) -> None:
        """Low variance returns CONTINUE action."""
        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(
                variance_threshold=0.15,
                prepare_ratio=0.5,
                embedding_dim=1024,
            ),
        )

        result = ssm.update("Test message")
        # With insufficient data, variance is 0, action is CONTINUE
        assert result.action == ActionState.CONTINUE

    def test_all_action_states_reachable(self) -> None:
        """All four action states can be reached based on thresholds."""
        # Test the internal action determination logic
        config = SSMConfig(
            variance_threshold=0.15,
            prepare_ratio=0.5,
            consolidate_ratio=2.0,
        )
        theta = config.variance_threshold

        # Create SSM to access _determine_action
        mock = MockEncoder()
        ssm = SemanticStateMonitor(encoder=mock, config=config)

        # CONTINUE: variance < 0.5 * theta
        assert ssm._determine_action(0.0) == ActionState.CONTINUE
        assert ssm._determine_action(0.07) == ActionState.CONTINUE

        # PREPARE: 0.5 * theta <= variance < theta
        assert ssm._determine_action(0.075) == ActionState.PREPARE
        assert ssm._determine_action(0.14) == ActionState.PREPARE

        # RETRIEVE: theta <= variance < 2 * theta
        assert ssm._determine_action(0.15) == ActionState.RETRIEVE
        assert ssm._determine_action(0.29) == ActionState.RETRIEVE

        # RETRIEVE_CONSOLIDATE: variance >= 2 * theta
        assert ssm._determine_action(0.30) == ActionState.RETRIEVE_CONSOLIDATE
        assert ssm._determine_action(1.0) == ActionState.RETRIEVE_CONSOLIDATE


class TestSSMWindowBuffer:
    """Test sliding window buffer behavior."""

    def test_window_buffer_is_fifo(self, mock_encoder: MockEncoder) -> None:
        """Window buffer operates as FIFO queue."""
        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(window_size=3, embedding_dim=1024),
        )

        # Fill beyond capacity
        for i in range(5):
            ssm.update(f"Message {i}")

        # Buffer should have exactly 3 items (last 3)
        assert len(ssm._buffer) == 3

    def test_window_size_matches_config(
        self, ssm: SemanticStateMonitor, ssm_config: SSMConfig
    ) -> None:
        """Window buffer respects configured size."""
        for i in range(20):
            ssm.update(f"Message {i}")

        assert len(ssm._buffer) == ssm_config.window_size


class TestSSMReset:
    """Test reset functionality."""

    def test_reset_clears_buffer(self, ssm: SemanticStateMonitor) -> None:
        """Reset clears embedding buffer."""
        for i in range(10):
            ssm.update(f"Message {i}")

        ssm.reset()
        assert len(ssm._buffer) == 0

    def test_reset_clears_turn_number(self, ssm: SemanticStateMonitor) -> None:
        """Reset resets turn number to 0."""
        for i in range(10):
            ssm.update(f"Message {i}")

        ssm.reset()
        assert ssm._turn_number == 0

    def test_reset_clears_smoothed_variance(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Reset resets smoothed variance to 0."""
        for i in range(10):
            ssm.update(f"Message {i}")

        ssm.reset()
        assert ssm._smoothed_variance == 0.0

    def test_reset_allows_fresh_start(self, ssm: SemanticStateMonitor) -> None:
        """After reset, SSM behaves like freshly initialized."""
        # Use once
        ssm.update("First conversation")
        ssm.update("Second message")

        # Reset
        ssm.reset()

        # Fresh start behavior
        result = ssm.update("New conversation")
        assert result.turn_number == 1
        assert result.window_size == 1
        assert result.variance == 0.0


class TestSSMErrorHandling:
    """Test error handling."""

    def test_empty_input_raises_ssm_error(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Empty string input raises SSMError."""
        with pytest.raises(SSMError):
            ssm.update("")

    def test_whitespace_input_raises_ssm_error(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Whitespace-only input raises SSMError."""
        with pytest.raises(SSMError):
            ssm.update("   \n\t  ")

    def test_dimension_mismatch_raises_encoding_error(
        self, mock_encoder: MockEncoder
    ) -> None:
        """Dimension mismatch between encoder and config raises EncodingError."""
        # Set up encoder to return wrong dimension
        mock_encoder._embedding_dim = 512
        wrong_emb = np.random.randn(512).astype(np.float32)
        mock_encoder.set_embeddings([wrong_emb])

        ssm = SemanticStateMonitor(
            encoder=mock_encoder,
            config=SSMConfig(embedding_dim=1024),  # Expects 1024
        )

        with pytest.raises(EncodingError, match="dimension mismatch"):
            ssm.update("Test")


class TestSSMObservability:
    """Test observability features."""

    def test_get_state_returns_expected_keys(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """get_state() returns all expected state keys."""
        ssm.update("Test message")
        state = ssm.get_state()

        expected_keys = {
            "turn_number",
            "window_size",
            "window_capacity",
            "smoothed_variance",
            "variance_threshold",
            "last_action",
        }
        assert set(state.keys()) == expected_keys

    def test_get_state_values_accurate(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """get_state() returns accurate current values."""
        ssm.update("Message 1")
        ssm.update("Message 2")

        state = ssm.get_state()

        assert state["turn_number"] == 2
        assert state["window_size"] == 2
        assert state["window_capacity"] == 5
        assert state["variance_threshold"] == 0.15
        assert state["last_action"] in [s.value for s in ActionState]


class TestSSMProperties:
    """Test SSM properties expose config correctly."""

    def test_window_size_property(
        self, ssm: SemanticStateMonitor, ssm_config: SSMConfig
    ) -> None:
        """window_size property matches config."""
        assert ssm.window_size == ssm_config.window_size

    def test_variance_threshold_property(
        self, ssm: SemanticStateMonitor, ssm_config: SSMConfig
    ) -> None:
        """variance_threshold property matches config."""
        assert ssm.variance_threshold == ssm_config.variance_threshold

    def test_smoothing_factor_property(
        self, ssm: SemanticStateMonitor, ssm_config: SSMConfig
    ) -> None:
        """smoothing_factor property matches config."""
        assert ssm.smoothing_factor == ssm_config.smoothing_factor


class TestSSMVarianceCalculation:
    """Test Ward variance calculation specifics."""

    def test_two_identical_embeddings_zero_variance(self) -> None:
        """Two identical embeddings have zero variance."""
        # Create fresh mock encoder for this test
        encoder = MockEncoder(embedding_dim=1024)
        emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        encoder.set_embeddings([emb.copy(), emb.copy()])

        ssm = SemanticStateMonitor(
            encoder=encoder,
            config=SSMConfig(embedding_dim=1024),
        )

        ssm.update("First")
        result = ssm.update("Second")

        # Identical embeddings should have zero variance
        assert result.variance == 0.0

    def test_mixed_embeddings_produces_variance(self) -> None:
        """Mix of similar and different embeddings produces non-zero variance.

        Note: With exactly 2 embeddings, Ward variance is always 0 because
        both points are equidistant from their centroid. We need 3+ points
        with varying distances from the centroid to get non-zero variance.
        """
        # Create fresh mock encoder for this test
        encoder = MockEncoder(embedding_dim=1024)

        # First two similar (close to each other)
        emb1 = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        emb2 = np.array([0.99, 0.1] + [0.0] * 1022, dtype=np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)  # Normalize

        # Third is orthogonal (far from first two)
        emb3 = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)

        encoder.set_embeddings([emb1, emb2, emb3])

        ssm = SemanticStateMonitor(
            encoder=encoder,
            config=SSMConfig(embedding_dim=1024),
        )

        ssm.update("Similar 1")
        ssm.update("Similar 2")
        result = ssm.update("Different")

        # With 3 points of varying distances from centroid, variance > 0
        assert result.variance > 0
