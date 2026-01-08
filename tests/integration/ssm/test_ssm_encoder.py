"""Integration tests for SSM with real Y-Encoder.

Tests validate that SSM correctly integrates with actual embedding models,
dimension validation works at runtime, and semantic boundary detection
functions with real embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

from prime.encoder import MINILM_CONFIG, YEncoder, YEncoderConfig
from prime.ssm import SSMConfig, SemanticStateMonitor
from prime.ssm.ssm_types import ActionState


@pytest.mark.integration
class TestSSMWithMiniLM:
    """Integration tests using MiniLM encoder (384 dims)."""

    @pytest.fixture(scope="class")
    def encoder(self) -> YEncoder:
        """MiniLM encoder instance shared across tests."""
        return YEncoder(MINILM_CONFIG)

    @pytest.fixture
    def ssm_config(self) -> SSMConfig:
        """SSM config matched to MiniLM dimensions."""
        return SSMConfig(
            embedding_dim=384,
            window_size=5,
            variance_threshold=0.15,
            smoothing_factor=0.3,
        )

    @pytest.fixture
    def ssm(self, encoder: YEncoder, ssm_config: SSMConfig) -> SemanticStateMonitor:
        """SSM instance with real encoder."""
        return SemanticStateMonitor(encoder, ssm_config)

    def test_ssm_update_returns_valid_result(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """SSM update with real encoder returns valid SemanticStateUpdate."""
        result = ssm.update("Hello, how are you?")

        assert result.variance >= 0.0
        assert result.smoothed_variance >= 0.0
        assert result.action in list(ActionState)
        assert result.window_size == 1
        assert result.turn_number == 1
        assert len(result.embedding) == 384

    def test_ssm_embedding_is_normalized(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Embedding in result is L2-normalized (from YEncoder)."""
        result = ssm.update("Test normalization")
        embedding = np.array(result.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_same_topic_continues(self, ssm: SemanticStateMonitor) -> None:
        """Similar messages on same topic yield CONTINUE state."""
        messages = [
            "Hello there",
            "Hi, how are you?",
            "Good morning!",
            "Hey, nice to meet you",
            "Greetings friend",
        ]

        results = [ssm.update(msg) for msg in messages]

        # With similar topics, variance should stay low
        # Most or all should be CONTINUE
        continue_count = sum(1 for r in results if r.action == ActionState.CONTINUE)
        assert continue_count >= 3, "Similar topics should mostly yield CONTINUE"

    def test_topic_change_increases_variance(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """Topic change increases variance compared to same topic."""
        # Build context with greeting messages
        ssm.update("Hello there")
        ssm.update("Hi, how are you?")
        ssm.update("Good morning!")
        ssm.update("Hey, nice to meet you")

        # Record variance before topic change
        result_same = ssm.update("Greetings friend")
        variance_same = result_same.smoothed_variance

        # Reset and test with topic change
        ssm.reset()
        ssm.update("Hello there")
        ssm.update("Hi, how are you?")
        ssm.update("Good morning!")
        ssm.update("Hey, nice to meet you")

        # Now introduce a topic change
        result_change = ssm.update("Explain quantum entanglement in physics")
        variance_change = result_change.smoothed_variance

        # Topic change should produce higher variance
        assert variance_change > variance_same

    def test_boundary_detection_on_major_shift(
        self, encoder: YEncoder
    ) -> None:
        """Major topic shift should increase variance vs same topic baseline."""
        # Low threshold to make variance differences detectable
        ssm_config = SSMConfig(
            embedding_dim=384,
            window_size=5,
            variance_threshold=0.01,  # Very low to detect small changes
            smoothing_factor=0.5,  # More responsive
        )
        ssm = SemanticStateMonitor(encoder, ssm_config)

        # Create context with code-related messages
        code_messages = [
            "How do I write a Python function?",
            "What is the syntax for a for loop in Python?",
            "Explain list comprehensions in Python",
            "How do I handle exceptions in Python?",
            "What are Python decorators?",
        ]

        for msg in code_messages:
            ssm.update(msg)

        # Major topic shift to completely different domain
        result_shift = ssm.update("What was the outcome of World War II?")

        # Reset and compare with same-topic continuation
        ssm.reset()
        for msg in code_messages:
            ssm.update(msg)

        # Same topic continuation
        result_same = ssm.update("How do I use Python lambda functions?")

        # Topic shift should produce higher variance than same topic
        # This is a relative comparison which is more stable than absolute thresholds
        assert result_shift.variance > result_same.variance, (
            f"Topic shift variance ({result_shift.variance:.6f}) should exceed "
            f"same-topic variance ({result_same.variance:.6f})"
        )

    def test_window_fills_correctly(self, ssm: SemanticStateMonitor) -> None:
        """Window buffer fills up to configured size."""
        messages = [f"Message number {i}" for i in range(7)]
        results = [ssm.update(msg) for msg in messages]

        # Window should cap at window_size (5)
        assert results[-1].window_size == 5

    def test_reset_clears_state(self, ssm: SemanticStateMonitor) -> None:
        """Reset clears all SSM state."""
        ssm.update("First message")
        ssm.update("Second message")
        ssm.update("Third message")

        ssm.reset()

        # After reset, first update should show turn_number=1
        result = ssm.update("New conversation")
        assert result.turn_number == 1
        assert result.window_size == 1

    def test_get_state_returns_observability_metrics(
        self, ssm: SemanticStateMonitor
    ) -> None:
        """get_state returns all required observability fields."""
        ssm.update("Test message")
        state = ssm.get_state()

        assert "turn_number" in state
        assert "window_size" in state
        assert "window_capacity" in state
        assert "smoothed_variance" in state
        assert "variance_threshold" in state
        assert "last_action" in state

        assert state["turn_number"] == 1
        assert state["window_size"] == 1
        assert state["window_capacity"] == 5


@pytest.mark.integration
class TestDimensionValidation:
    """Tests for runtime dimension validation."""

    def test_dimension_mismatch_raises_encoding_error(self) -> None:
        """Mismatched SSM config dimension raises EncodingError on update."""
        from prime.ssm.exceptions import EncodingError

        # Encoder produces 384-dim, but SSM expects 1024
        encoder = YEncoder(MINILM_CONFIG)
        ssm_config = SSMConfig(embedding_dim=1024)
        ssm = SemanticStateMonitor(encoder, ssm_config)

        with pytest.raises(EncodingError, match="dimension mismatch"):
            ssm.update("Test message")

    def test_matched_dimensions_work_correctly(self) -> None:
        """Correctly matched dimensions work without error."""
        encoder = YEncoder(MINILM_CONFIG)
        ssm_config = SSMConfig(embedding_dim=384)
        ssm = SemanticStateMonitor(encoder, ssm_config)

        result = ssm.update("Test message")
        assert len(result.embedding) == 384


@pytest.mark.integration
class TestEncoderProtocolCompliance:
    """Tests that SSM works with any Encoder protocol implementation."""

    def test_yencoder_satisfies_encoder_protocol(self) -> None:
        """YEncoder satisfies the Encoder protocol at runtime."""
        from prime.encoder.protocols import Encoder

        encoder = YEncoder(MINILM_CONFIG)
        assert isinstance(encoder, Encoder)

    def test_ssm_accepts_encoder_protocol(self) -> None:
        """SSM accepts any Encoder protocol implementation."""
        encoder = YEncoder(MINILM_CONFIG)
        ssm_config = SSMConfig(embedding_dim=384)

        # Should work with YEncoder implementing Encoder protocol
        ssm = SemanticStateMonitor(encoder, ssm_config)
        result = ssm.update("Protocol test")

        assert result.turn_number == 1


@pytest.mark.integration
class TestSemanticQuality:
    """Tests verifying semantic quality of boundary detection."""

    @pytest.fixture(scope="class")
    def encoder(self) -> YEncoder:
        """Shared encoder for semantic tests."""
        return YEncoder(MINILM_CONFIG)

    def test_gradual_drift_detected(self, encoder: YEncoder) -> None:
        """Gradual topic drift eventually triggers boundary."""
        ssm_config = SSMConfig(
            embedding_dim=384,
            window_size=5,
            variance_threshold=0.1,  # Lower threshold for drift detection
        )
        ssm = SemanticStateMonitor(encoder, ssm_config)

        # Start with Python topic
        ssm.update("Python is a programming language")
        ssm.update("Python has dynamic typing")
        ssm.update("Python supports multiple paradigms")

        # Drift to data science
        ssm.update("Python is used for data science")
        ssm.update("Machine learning uses Python")

        # Major drift to unrelated topic
        result = ssm.update("How to bake a chocolate cake?")

        # Should detect this as significant variance increase
        assert result.variance > 0.01 or result.action != ActionState.CONTINUE

    def test_coherent_conversation_stays_continue(
        self, encoder: YEncoder
    ) -> None:
        """Coherent conversation on single topic stays in CONTINUE."""
        ssm_config = SSMConfig(
            embedding_dim=384,
            window_size=5,
            variance_threshold=0.3,  # Higher threshold for stricter boundary
        )
        ssm = SemanticStateMonitor(encoder, ssm_config)

        # All about cooking
        cooking_messages = [
            "I want to learn cooking",
            "What ingredients do I need for pasta?",
            "How long should I boil the pasta?",
            "What sauce goes well with spaghetti?",
            "Should I add parmesan cheese?",
            "How do I make the sauce from scratch?",
        ]

        results = [ssm.update(msg) for msg in cooking_messages]

        # Most should be CONTINUE due to coherent topic
        continue_count = sum(1 for r in results if r.action == ActionState.CONTINUE)
        assert continue_count >= 4, "Coherent topic should yield mostly CONTINUE"


@pytest.mark.integration
@pytest.mark.slow
class TestLargerEmbeddingModel:
    """Tests with larger embedding models (when available)."""

    @pytest.fixture(scope="class")
    def bge_encoder(self) -> YEncoder:
        """BGE-large encoder (1024 dims)."""
        from prime.encoder import BGE_LARGE_CONFIG

        return YEncoder(BGE_LARGE_CONFIG)

    def test_ssm_with_bge_large(self, bge_encoder: YEncoder) -> None:
        """SSM works with BGE-large 1024-dim embeddings."""
        ssm_config = SSMConfig(
            embedding_dim=1024,
            window_size=5,
            variance_threshold=0.15,
        )
        ssm = SemanticStateMonitor(bge_encoder, ssm_config)

        result = ssm.update("Test with larger model")
        assert len(result.embedding) == 1024
        assert result.turn_number == 1
