"""Unit tests for PRIME API middleware.

Tests for authentication, rate limiting, and logging middleware.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prime.api.middleware import (
    APIKeyMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# Test fixtures


def create_test_app() -> FastAPI:
    """Create a minimal FastAPI app for testing middleware."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health")
    async def health_endpoint() -> dict[str, str]:
        return {"status": "healthy"}

    return app


class TestAPIKeyMiddleware:
    """Tests for API key authentication middleware."""

    def test_disabled_middleware_allows_all(self) -> None:
        """Test that disabled middleware allows all requests."""
        app = create_test_app()
        app.add_middleware(APIKeyMiddleware, api_keys=frozenset({"key1"}), enabled=False)

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200

    def test_no_keys_disables_auth(self) -> None:
        """Test that empty API keys set disables authentication."""
        app = create_test_app()
        app.add_middleware(APIKeyMiddleware, api_keys=frozenset(), enabled=True)

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200

    def test_missing_api_key_returns_401(self) -> None:
        """Test that missing API key returns 401."""
        app = create_test_app()
        app.add_middleware(APIKeyMiddleware, api_keys=frozenset({"valid-key"}))

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 401
            data = response.json()
            assert data["error_code"] == "MISSING_API_KEY"

    def test_invalid_api_key_returns_401(self) -> None:
        """Test that invalid API key returns 401."""
        app = create_test_app()
        app.add_middleware(APIKeyMiddleware, api_keys=frozenset({"valid-key"}))

        with TestClient(app) as client:
            response = client.get("/test", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401
            data = response.json()
            assert data["error_code"] == "INVALID_API_KEY"

    def test_valid_api_key_allows_request(self) -> None:
        """Test that valid API key allows request."""
        app = create_test_app()
        app.add_middleware(APIKeyMiddleware, api_keys=frozenset({"valid-key"}))

        with TestClient(app) as client:
            response = client.get("/test", headers={"X-API-Key": "valid-key"})
            assert response.status_code == 200

    def test_public_path_bypasses_auth(self) -> None:
        """Test that public paths bypass authentication."""
        app = create_test_app()
        app.add_middleware(APIKeyMiddleware, api_keys=frozenset({"valid-key"}))

        with TestClient(app) as client:
            response = client.get("/api/v1/health")
            assert response.status_code == 200

    def test_multiple_valid_keys(self) -> None:
        """Test that multiple valid keys all work."""
        app = create_test_app()
        app.add_middleware(
            APIKeyMiddleware, api_keys=frozenset({"key1", "key2", "key3"})
        )

        with TestClient(app) as client:
            for key in ["key1", "key2", "key3"]:
                response = client.get("/test", headers={"X-API-Key": key})
                assert response.status_code == 200


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    def test_disabled_middleware_allows_all(self) -> None:
        """Test that disabled middleware allows all requests."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=1, window_seconds=60, enabled=False
        )

        with TestClient(app) as client:
            for _ in range(5):
                response = client.get("/test")
                assert response.status_code == 200

    def test_within_rate_limit(self) -> None:
        """Test requests within rate limit succeed."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=10, window_seconds=60, enabled=True
        )

        with TestClient(app) as client:
            for _ in range(5):
                response = client.get("/test")
                assert response.status_code == 200
                assert "X-RateLimit-Remaining" in response.headers

    def test_exceeding_rate_limit_returns_429(self) -> None:
        """Test that exceeding rate limit returns 429."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=3, window_seconds=60, enabled=True
        )

        with TestClient(app) as client:
            # First 3 requests should succeed
            for _ in range(3):
                response = client.get("/test")
                assert response.status_code == 200

            # 4th request should fail
            response = client.get("/test")
            assert response.status_code == 429
            data = response.json()
            assert data["error_code"] == "RATE_LIMIT_EXCEEDED"
            assert "Retry-After" in response.headers

    def test_rate_limit_headers(self) -> None:
        """Test that rate limit headers are present in response."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=10, window_seconds=60, enabled=True
        )

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers
            assert response.headers["X-RateLimit-Limit"] == "10"

    def test_rate_limit_by_api_key(self) -> None:
        """Test that rate limiting uses API key when present."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=2, window_seconds=60, enabled=True
        )

        with TestClient(app) as client:
            # Exhaust rate limit for key1
            for _ in range(2):
                response = client.get("/test", headers={"X-API-Key": "key1"})
                assert response.status_code == 200

            # key1 should be rate limited
            response = client.get("/test", headers={"X-API-Key": "key1"})
            assert response.status_code == 429

            # key2 should still work (different client)
            response = client.get("/test", headers={"X-API-Key": "key2"})
            assert response.status_code == 200

    def test_rate_limit_by_ip(self) -> None:
        """Test that rate limiting uses IP when no API key."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=2, window_seconds=60, enabled=True
        )

        with TestClient(app) as client:
            # Exhaust rate limit
            for _ in range(2):
                response = client.get("/test")
                assert response.status_code == 200

            # Should be rate limited
            response = client.get("/test")
            assert response.status_code == 429

    def test_rate_limit_window_expiry(self) -> None:
        """Test that rate limit resets after window expires."""
        app = create_test_app()
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=2, window_seconds=1, enabled=True
        )

        with TestClient(app) as client:
            # Exhaust rate limit
            for _ in range(2):
                response = client.get("/test")
                assert response.status_code == 200

            # Wait for window to expire
            time.sleep(1.1)

            # Should be allowed again
            response = client.get("/test")
            assert response.status_code == 200


class TestLoggingMiddleware:
    """Tests for logging middleware."""

    def test_adds_request_id_header(self) -> None:
        """Test that X-Request-ID header is added."""
        app = create_test_app()
        app.add_middleware(LoggingMiddleware)

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200
            assert "X-Request-ID" in response.headers
            # UUID format check
            request_id = response.headers["X-Request-ID"]
            assert len(request_id) == 36
            assert request_id.count("-") == 4

    def test_adds_timing_header(self) -> None:
        """Test that X-Request-Time-Ms header is added."""
        app = create_test_app()
        app.add_middleware(LoggingMiddleware)

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200
            assert "X-Request-Time-Ms" in response.headers
            # Should be a number
            time_ms = float(response.headers["X-Request-Time-Ms"])
            assert time_ms >= 0

    def test_unique_request_ids(self) -> None:
        """Test that each request gets a unique ID."""
        app = create_test_app()
        app.add_middleware(LoggingMiddleware)

        request_ids: set[str] = set()
        with TestClient(app) as client:
            for _ in range(10):
                response = client.get("/test")
                request_id = response.headers["X-Request-ID"]
                assert request_id not in request_ids
                request_ids.add(request_id)

    def test_logs_request_start(self) -> None:
        """Test that request start is logged."""
        app = create_test_app()
        app.add_middleware(LoggingMiddleware)

        with patch("prime.api.middleware.logging.logger") as mock_logger:
            with TestClient(app) as client:
                client.get("/test")
                mock_logger.debug.assert_called()
                call_args = mock_logger.debug.call_args
                assert call_args[0][0] == "request_started"

    def test_logs_request_completion(self) -> None:
        """Test that request completion is logged."""
        app = create_test_app()
        app.add_middleware(LoggingMiddleware)

        with patch("prime.api.middleware.logging.logger") as mock_logger:
            with TestClient(app) as client:
                client.get("/test")
                mock_logger.info.assert_called()
                call_args = mock_logger.info.call_args
                assert call_args[0][0] == "request_completed"


class TestMiddlewareCombination:
    """Tests for combined middleware behavior."""

    @pytest.fixture
    def full_app(self) -> Iterator[TestClient]:
        """Create app with all middleware enabled."""
        app = create_test_app()
        # Order matters: last added is first to process
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=100, window_seconds=60, enabled=True
        )
        app.add_middleware(
            APIKeyMiddleware, api_keys=frozenset({"test-key"}), enabled=True
        )

        with TestClient(app) as client:
            yield client

    def test_all_headers_present_on_success(self, full_app: TestClient) -> None:
        """Test that all middleware headers are present on success."""
        response = full_app.get("/api/v1/health")  # Public path
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Request-Time-Ms" in response.headers
        assert "X-RateLimit-Limit" in response.headers

    def test_auth_failure_stops_pipeline(self, full_app: TestClient) -> None:
        """Test that auth failure prevents further processing."""
        response = full_app.get("/test")  # No API key
        assert response.status_code == 401
        # Rate limit should not have been checked/incremented

    def test_full_pipeline_success(self, full_app: TestClient) -> None:
        """Test successful request through full middleware pipeline."""
        response = full_app.get("/test", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
