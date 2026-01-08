"""Unit tests for PRIME configuration."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from pydantic import ValidationError

from prime import (
    APIConfig,
    PRIMEConfig,
    RAGASConfig,
)
from prime.exceptions import ConfigurationError


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_values(self) -> None:
        """Test APIConfig default values."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4
        assert config.api_key_header == "X-API-Key"
        assert config.rate_limit_per_minute == 60
        assert config.cors_origins == ["*"]
        assert config.request_timeout_seconds == 30.0

    def test_custom_values(self) -> None:
        """Test APIConfig with custom values."""
        config = APIConfig(
            host="127.0.0.1",
            port=9000,
            workers=8,
            api_key_header="Authorization",
            rate_limit_per_minute=120,
            cors_origins=["https://example.com"],
            request_timeout_seconds=60.0,
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.workers == 8
        assert config.api_key_header == "Authorization"
        assert config.rate_limit_per_minute == 120
        assert config.cors_origins == ["https://example.com"]
        assert config.request_timeout_seconds == 60.0

    def test_port_validation_min(self) -> None:
        """Test port validation rejects values below 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIConfig(port=0)

    def test_port_validation_max(self) -> None:
        """Test port validation rejects values above 65535."""
        with pytest.raises(ValidationError, match="less than or equal to 65535"):
            APIConfig(port=70000)

    def test_workers_validation_min(self) -> None:
        """Test workers validation rejects values below 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIConfig(workers=0)

    def test_workers_validation_max(self) -> None:
        """Test workers validation rejects values above 32."""
        with pytest.raises(ValidationError, match="less than or equal to 32"):
            APIConfig(workers=64)

    def test_frozen_model(self) -> None:
        """Test APIConfig is immutable."""
        config = APIConfig()
        with pytest.raises(ValidationError):
            config.port = 9000  # type: ignore[misc]


class TestRAGASConfig:
    """Tests for RAGASConfig."""

    def test_default_values(self) -> None:
        """Test RAGASConfig default values."""
        config = RAGASConfig()
        assert config.enabled is True
        assert config.llm_model == "gpt-4.1-mini"
        assert config.batch_size == 10
        assert config.timeout_seconds == 60.0

    def test_custom_values(self) -> None:
        """Test RAGASConfig with custom values."""
        config = RAGASConfig(
            enabled=False,
            llm_model="gpt-4.1",
            batch_size=20,
            timeout_seconds=120.0,
        )
        assert config.enabled is False
        assert config.llm_model == "gpt-4.1"
        assert config.batch_size == 20
        assert config.timeout_seconds == 120.0

    def test_batch_size_validation_min(self) -> None:
        """Test batch_size validation rejects values below 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            RAGASConfig(batch_size=0)

    def test_batch_size_validation_max(self) -> None:
        """Test batch_size validation rejects values above 100."""
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            RAGASConfig(batch_size=200)

    def test_timeout_validation(self) -> None:
        """Test timeout_seconds validation rejects values below 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            RAGASConfig(timeout_seconds=0.5)


class TestPRIMEConfig:
    """Tests for PRIMEConfig."""

    def test_default_values(self) -> None:
        """Test PRIMEConfig default values."""
        config = PRIMEConfig()
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3.5-sonnet"
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.ragas, RAGASConfig)

    def test_nested_configs(self) -> None:
        """Test PRIMEConfig contains nested component configs."""
        config = PRIMEConfig()
        # Check SSM config
        assert config.ssm.window_size == 5
        assert config.ssm.variance_threshold == 0.15
        # Check MCS config
        assert config.mcs.similarity_threshold == 0.85
        # Check Predictor config
        assert config.predictor.hidden_dim == 2048
        # Check Y-Encoder config
        assert config.y_encoder.normalize is True

    def test_custom_nested_configs(self) -> None:
        """Test PRIMEConfig with custom nested configs."""
        from prime.encoder.config import YEncoderConfig
        from prime.mcs.mcs_config import MCSConfig
        from prime.predictor.config import PredictorConfig
        from prime.ssm.ssm_config import SSMConfig

        config = PRIMEConfig(
            ssm=SSMConfig(window_size=10),
            mcs=MCSConfig(similarity_threshold=0.90),
            predictor=PredictorConfig(hidden_dim=1024, num_heads=4),
            y_encoder=YEncoderConfig(embedding_dim=512),
        )
        assert config.ssm.window_size == 10
        assert config.mcs.similarity_threshold == 0.90
        assert config.predictor.hidden_dim == 1024
        assert config.y_encoder.embedding_dim == 512

    def test_from_env_defaults(self) -> None:
        """Test from_env with no environment variables."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = PRIMEConfig.from_env()
            assert config.api.host == "0.0.0.0"
            assert config.api.port == 8000
            assert config.ragas.enabled is True

    def test_from_env_custom_values(self) -> None:
        """Test from_env with custom environment variables."""
        env_vars = {
            "PRIME_HOST": "127.0.0.1",
            "PRIME_PORT": "9000",
            "PRIME_WORKERS": "8",
            "PRIME_RATE_LIMIT": "100",
            "PRIME_RAGAS_ENABLED": "false",
            "PRIME_RAGAS_MODEL": "gpt-4.1",
            "QDRANT_HOST": "qdrant.example.com",
            "QDRANT_PORT": "6334",
            "PRIME_LLM_PROVIDER": "openai",
            "PRIME_LLM_MODEL": "gpt-4.1",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            config = PRIMEConfig.from_env()
            assert config.api.host == "127.0.0.1"
            assert config.api.port == 9000
            assert config.api.workers == 8
            assert config.api.rate_limit_per_minute == 100
            assert config.ragas.enabled is False
            assert config.ragas.llm_model == "gpt-4.1"
            assert config.mcs.qdrant_host == "qdrant.example.com"
            assert config.mcs.qdrant_port == 6334
            assert config.llm_provider == "openai"
            assert config.llm_model == "gpt-4.1"

    def test_for_testing(self) -> None:
        """Test for_testing creates test-friendly config."""
        config = PRIMEConfig.for_testing()
        # Should use smaller models
        assert config.y_encoder.embedding_dim == 384
        assert config.ssm.embedding_dim == 384
        assert config.mcs.embedding_dim == 384
        assert config.predictor.input_dim == 384
        assert config.predictor.output_dim == 384
        # Should use FAISS instead of Qdrant
        assert config.mcs.index_type == "faiss"
        # Should disable RAGAS
        assert config.ragas.enabled is False

    def test_validate_for_production_valid(self) -> None:
        """Test validate_for_production passes with matching dimensions."""
        config = PRIMEConfig.for_testing()
        # This should not raise
        config.validate_for_production()

    def test_validate_for_production_dimension_mismatch(self) -> None:
        """Test validate_for_production fails with mismatched dimensions."""
        from prime.encoder.config import YEncoderConfig
        from prime.ssm.ssm_config import SSMConfig

        config = PRIMEConfig(
            y_encoder=YEncoderConfig(embedding_dim=1024),
            ssm=SSMConfig(embedding_dim=384),  # Mismatch!
        )
        with pytest.raises(ConfigurationError, match="Y-Encoder embedding_dim"):
            config.validate_for_production()


class TestExceptions:
    """Tests for PRIME exception hierarchy."""

    def test_prime_error_base(self) -> None:
        """Test PRIMEError base exception."""
        from prime import PRIMEError

        error = PRIMEError("Test error", "TEST_CODE")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "TEST_CODE"

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert error.error_code == "PRIME_CONFIG_ERROR"

    def test_component_error(self) -> None:
        """Test ComponentError."""
        from prime import ComponentError

        error = ComponentError("mcs", "Connection failed")
        assert str(error) == "mcs: Connection failed"
        assert error.error_code == "PRIME_MCS_ERROR"
        assert error.component == "mcs"

    def test_session_error(self) -> None:
        """Test SessionError."""
        from prime import SessionError

        error = SessionError("sess_123", "Session expired")
        assert str(error) == "Session sess_123: Session expired"
        assert error.error_code == "PRIME_SESSION_ERROR"
        assert error.session_id == "sess_123"

    def test_authentication_error_default(self) -> None:
        """Test AuthenticationError with default message."""
        from prime import AuthenticationError

        error = AuthenticationError()
        assert str(error) == "Authentication required"
        assert error.error_code == "PRIME_AUTH_ERROR"

    def test_authentication_error_custom(self) -> None:
        """Test AuthenticationError with custom message."""
        from prime import AuthenticationError

        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError."""
        from prime import RateLimitError

        error = RateLimitError(60)
        assert str(error) == "Rate limit exceeded. Retry after 60s"
        assert error.error_code == "PRIME_RATE_LIMIT_ERROR"
        assert error.retry_after == 60


class TestTypes:
    """Tests for PRIME type definitions."""

    def test_action_state_values(self) -> None:
        """Test ActionState enum values."""
        from prime import ActionState

        assert ActionState.CONTINUE == "continue"
        assert ActionState.PREPARE == "prepare"
        assert ActionState.RETRIEVE == "retrieve"
        assert ActionState.RETRIEVE_CONSOLIDATE == "retrieve_consolidate"

    def test_memory_read_result(self) -> None:
        """Test MemoryReadResult dataclass."""
        from prime import MemoryReadResult

        result = MemoryReadResult(
            memory_id="mem_123",
            content="Test content",
            cluster_id=1,
            similarity=0.95,
            metadata={"source": "test"},
            created_at=1234567890.0,
        )
        assert result.memory_id == "mem_123"
        assert result.content == "Test content"
        assert result.cluster_id == 1
        assert result.similarity == 0.95
        assert result.metadata == {"source": "test"}
        assert result.created_at == 1234567890.0

    def test_memory_read_result_defaults(self) -> None:
        """Test MemoryReadResult default values."""
        from prime import MemoryReadResult

        result = MemoryReadResult(
            memory_id="mem_123",
            content="Test",
            cluster_id=1,
            similarity=0.9,
        )
        assert result.metadata == {}
        assert result.created_at == 0.0

    def test_memory_write_result(self) -> None:
        """Test MemoryWriteResult dataclass."""
        from prime import MemoryWriteResult

        result = MemoryWriteResult(
            memory_id="mem_456",
            cluster_id=2,
            is_new_cluster=True,
            consolidated=False,
        )
        assert result.memory_id == "mem_456"
        assert result.cluster_id == 2
        assert result.is_new_cluster is True
        assert result.consolidated is False

    def test_component_status(self) -> None:
        """Test ComponentStatus dataclass."""
        from prime import ComponentStatus

        status = ComponentStatus(
            name="ssm",
            status="healthy",
            latency_p50_ms=15.5,
            error_rate=0.001,
        )
        assert status.name == "ssm"
        assert status.status == "healthy"
        assert status.latency_p50_ms == 15.5
        assert status.error_rate == 0.001

    def test_prime_response(self) -> None:
        """Test PRIMEResponse dataclass."""
        from prime import ActionState, MemoryReadResult, PRIMEResponse

        memory = MemoryReadResult(
            memory_id="mem_1",
            content="Test",
            cluster_id=1,
            similarity=0.9,
        )
        response = PRIMEResponse(
            retrieved_memories=[memory],
            boundary_crossed=True,
            variance=0.2,
            smoothed_variance=0.18,
            action=ActionState.RETRIEVE,
            session_id="sess_123",
            turn_number=5,
            latency_ms=45.2,
        )
        assert len(response.retrieved_memories) == 1
        assert response.boundary_crossed is True
        assert response.variance == 0.2
        assert response.smoothed_variance == 0.18
        assert response.action == ActionState.RETRIEVE
        assert response.session_id == "sess_123"
        assert response.turn_number == 5
        assert response.latency_ms == 45.2

    def test_prime_diagnostics(self) -> None:
        """Test PRIMEDiagnostics dataclass."""
        from prime import ComponentStatus, PRIMEDiagnostics

        ssm_status = ComponentStatus(
            name="ssm",
            status="healthy",
            latency_p50_ms=10.0,
            error_rate=0.0,
        )
        diagnostics = PRIMEDiagnostics(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
            components={"ssm": ssm_status},
            metrics={"total_requests": 1000.0},
        )
        assert diagnostics.status == "healthy"
        assert diagnostics.version == "1.0.0"
        assert diagnostics.uptime_seconds == 3600.0
        assert "ssm" in diagnostics.components
        assert diagnostics.metrics["total_requests"] == 1000.0

    def test_dataclass_frozen(self) -> None:
        """Test dataclasses are immutable."""
        from prime import MemoryReadResult

        result = MemoryReadResult(
            memory_id="mem_1",
            content="Test",
            cluster_id=1,
            similarity=0.9,
        )
        with pytest.raises(AttributeError):
            result.memory_id = "new_id"  # type: ignore[misc]
