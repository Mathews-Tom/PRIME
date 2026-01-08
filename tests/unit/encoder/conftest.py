"""Shared fixtures for encoder tests."""

from __future__ import annotations

import pytest

from prime.encoder import MINILM_CONFIG, YEncoder, YEncoderConfig


@pytest.fixture(scope="module")
def encoder_config() -> YEncoderConfig:
    """MiniLM configuration for fast testing."""
    return MINILM_CONFIG


@pytest.fixture(scope="module")
def encoder(encoder_config: YEncoderConfig) -> YEncoder:
    """Shared YEncoder instance using MiniLM.

    Module-scoped to avoid repeated model loading.
    """
    return YEncoder(encoder_config)
