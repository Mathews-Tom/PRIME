"""PRIME configuration.

Defines the main PRIMEConfig class that aggregates all component
configurations and provides environment-based loading.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from prime.encoder.config import YEncoderConfig
from prime.exceptions import ConfigurationError
from prime.mcs.mcs_config import MCSConfig
from prime.predictor.config import PredictorConfig
from prime.ssm.ssm_config import SSMConfig


class APIConfig(BaseModel):
    """API server configuration.

    Controls FastAPI server settings including host, port,
    authentication, rate limiting, and CORS.

    Attributes:
        host: Server bind address.
        port: Server port number.
        workers: Number of uvicorn workers.
        api_key_header: Header name for API key authentication.
        rate_limit_per_minute: Maximum requests per minute per client.
        cors_origins: Allowed CORS origins.
        request_timeout_seconds: Maximum request processing time.
    """

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=60, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = Field(default=30.0, ge=1.0)

    model_config = {"frozen": True}


class RAGASConfig(BaseModel):
    """RAGAS evaluation configuration.

    Controls the RAGAS evaluation system for measuring
    RAG quality metrics.

    Attributes:
        enabled: Whether RAGAS evaluation is enabled.
        llm_model: LLM model for RAGAS evaluations.
        batch_size: Batch size for evaluation.
        timeout_seconds: Evaluation timeout.
    """

    enabled: bool = Field(default=True)
    llm_model: str = Field(default="gpt-4.1-mini")
    batch_size: int = Field(default=10, ge=1, le=100)
    timeout_seconds: float = Field(default=60.0, ge=1.0)

    model_config = {"frozen": True}


class PRIMEConfig(BaseModel):
    """Main PRIME system configuration.

    Aggregates all component configurations into a single
    immutable configuration object. Supports loading from
    environment variables.

    Attributes:
        ssm: Semantic State Monitor configuration.
        mcs: Memory Cluster Store configuration.
        predictor: Embedding Predictor configuration.
        y_encoder: Y-Encoder configuration.
        api: API server configuration.
        ragas: RAGAS evaluation configuration.
        llm_provider: LLM provider for response generation.
        llm_model: LLM model name.
    """

    # Component configs
    ssm: SSMConfig = Field(default_factory=SSMConfig)
    mcs: MCSConfig = Field(default_factory=MCSConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    y_encoder: YEncoderConfig = Field(default_factory=YEncoderConfig)

    # API config
    api: APIConfig = Field(default_factory=APIConfig)

    # Evaluation config
    ragas: RAGASConfig = Field(default_factory=RAGASConfig)

    # LLM integration
    llm_provider: str = Field(default="anthropic")
    llm_model: str = Field(default="claude-3.5-sonnet")

    model_config = {"frozen": True}

    @classmethod
    def from_env(cls) -> PRIMEConfig:
        """Load configuration from environment variables.

        Creates a PRIMEConfig with values loaded from environment
        variables, using defaults where not specified.

        Environment Variables:
            PRIME_HOST: API server host (default: 0.0.0.0)
            PRIME_PORT: API server port (default: 8000)
            PRIME_WORKERS: Number of workers (default: 4)
            PRIME_RATE_LIMIT: Requests per minute (default: 60)
            PRIME_RAGAS_ENABLED: Enable RAGAS (default: true)
            PRIME_RAGAS_MODEL: RAGAS LLM model (default: gpt-4.1-mini)
            QDRANT_URL: Qdrant server URL (required for production)
            QDRANT_API_KEY: Qdrant API key (optional)

        Returns:
            PRIMEConfig with values from environment.

        Raises:
            ConfigurationError: If required variables missing in production.
        """
        # Build API config from environment
        api_config = APIConfig(
            host=os.getenv("PRIME_HOST", "0.0.0.0"),
            port=int(os.getenv("PRIME_PORT", "8000")),
            workers=int(os.getenv("PRIME_WORKERS", "4")),
            rate_limit_per_minute=int(os.getenv("PRIME_RATE_LIMIT", "60")),
        )

        # Build RAGAS config from environment
        ragas_config = RAGASConfig(
            enabled=os.getenv("PRIME_RAGAS_ENABLED", "true").lower() == "true",
            llm_model=os.getenv("PRIME_RAGAS_MODEL", "gpt-4.1-mini"),
        )

        # Build MCS config with Qdrant settings from environment
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        mcs_config = MCSConfig(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key,
        )

        # Build Y-Encoder config from environment
        model_name = os.getenv(
            "PRIME_ENCODER_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        y_encoder_config = YEncoderConfig(model_name=model_name)

        # LLM settings
        llm_provider = os.getenv("PRIME_LLM_PROVIDER", "anthropic")
        llm_model = os.getenv("PRIME_LLM_MODEL", "claude-3.5-sonnet")

        return cls(
            api=api_config,
            ragas=ragas_config,
            mcs=mcs_config,
            y_encoder=y_encoder_config,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    @classmethod
    def for_testing(cls) -> PRIMEConfig:
        """Create configuration optimized for testing.

        Returns a PRIMEConfig with settings suitable for unit
        and integration tests, using smaller models and disabled
        external services.

        Returns:
            PRIMEConfig for testing environments.
        """
        return cls(
            y_encoder=YEncoderConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=384,
            ),
            ssm=SSMConfig(embedding_dim=384),
            mcs=MCSConfig(embedding_dim=384, index_type="faiss"),
            predictor=PredictorConfig(
                input_dim=384,
                hidden_dim=768,
                output_dim=384,
                num_layers=2,
                num_heads=4,
            ),
            ragas=RAGASConfig(enabled=False),
        )

    def validate_for_production(self) -> None:
        """Validate configuration for production deployment.

        Checks that all required settings are properly configured
        for production use.

        Raises:
            ConfigurationError: If configuration is invalid for production.
        """
        errors: list[str] = []

        # Check embedding dimensions match across components
        if self.y_encoder.embedding_dim != self.ssm.embedding_dim:
            errors.append(
                f"Y-Encoder embedding_dim ({self.y_encoder.embedding_dim}) "
                f"!= SSM embedding_dim ({self.ssm.embedding_dim})"
            )

        if self.y_encoder.embedding_dim != self.mcs.embedding_dim:
            errors.append(
                f"Y-Encoder embedding_dim ({self.y_encoder.embedding_dim}) "
                f"!= MCS embedding_dim ({self.mcs.embedding_dim})"
            )

        if self.y_encoder.embedding_dim != self.predictor.input_dim:
            errors.append(
                f"Y-Encoder embedding_dim ({self.y_encoder.embedding_dim}) "
                f"!= Predictor input_dim ({self.predictor.input_dim})"
            )

        if self.y_encoder.embedding_dim != self.predictor.output_dim:
            errors.append(
                f"Y-Encoder embedding_dim ({self.y_encoder.embedding_dim}) "
                f"!= Predictor output_dim ({self.predictor.output_dim})"
            )

        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
