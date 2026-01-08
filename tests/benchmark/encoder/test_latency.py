"""Latency benchmark tests for YEncoder.

These tests measure encoding latency and validate performance targets.
Run with: uv run pytest tests/benchmark/ -v -m benchmark
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from prime.encoder import MINILM_CONFIG, YEncoder, YEncoderConfig


@pytest.fixture(scope="module")
def encoder() -> YEncoder:
    """MiniLM encoder for benchmarking."""
    return YEncoder(MINILM_CONFIG)


@pytest.fixture(scope="module")
def cached_encoder() -> YEncoder:
    """MiniLM encoder with caching enabled."""
    config = YEncoderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        cache_size=1000,
    )
    return YEncoder(config)


@pytest.mark.benchmark
class TestSingleEncodingLatency:
    """Benchmark tests for single text encoding."""

    def test_single_encode_produces_output(self, encoder: YEncoder) -> None:
        """Single encoding produces correct output."""
        embedding = encoder.encode("Test text for benchmarking")
        assert embedding.shape == (encoder.embedding_dim,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_single_encode_latency_measurement(self, encoder: YEncoder) -> None:
        """Measure single encoding latency."""
        text = "This is a test sentence for latency measurement."

        # Warmup
        encoder.encode(text)

        # Measure
        iterations = 10
        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            encoder.encode(text)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_ms = np.mean(times)
        p50_ms = np.percentile(times, 50)
        p95_ms = np.percentile(times, 95)

        # Report (always passes, just measures)
        print("\nSingle encode latency:")
        print(f"  Average: {avg_ms:.2f}ms")
        print(f"  p50: {p50_ms:.2f}ms")
        print(f"  p95: {p95_ms:.2f}ms")

        # Soft assertion for CI (generous limit for CPU)
        assert p50_ms < 500, f"p50 latency {p50_ms:.2f}ms exceeds 500ms threshold"


@pytest.mark.benchmark
class TestBatchEncodingLatency:
    """Benchmark tests for batch encoding."""

    def test_batch_encode_produces_output(self, encoder: YEncoder) -> None:
        """Batch encoding produces correct output."""
        texts = [f"Test text {i}" for i in range(32)]
        embeddings = encoder.encode_batch(texts)
        assert len(embeddings) == 32
        for emb in embeddings:
            assert emb.shape == (encoder.embedding_dim,)

    def test_batch_encode_latency_measurement(self, encoder: YEncoder) -> None:
        """Measure batch encoding latency."""
        texts = [f"Sentence number {i} for batch encoding test." for i in range(32)]

        # Warmup
        encoder.encode_batch(texts)

        # Measure
        iterations = 5
        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            encoder.encode_batch(texts)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_ms = np.mean(times)
        p50_ms = np.percentile(times, 50)
        p95_ms = np.percentile(times, 95)

        print("\nBatch encode (32) latency:")
        print(f"  Average: {avg_ms:.2f}ms")
        print(f"  p50: {p50_ms:.2f}ms")
        print(f"  p95: {p95_ms:.2f}ms")
        print(f"  Per-text: {avg_ms/32:.2f}ms")

        # Soft assertion (generous for CPU)
        assert p50_ms < 2000, f"p50 latency {p50_ms:.2f}ms exceeds 2000ms threshold"


@pytest.mark.benchmark
class TestCachePerformance:
    """Benchmark tests for caching performance."""

    def test_cache_hit_faster_than_miss(self, cached_encoder: YEncoder) -> None:
        """Cache hits should be significantly faster than misses."""
        text = "Text for cache performance testing"

        # First encode (miss)
        cached_encoder.clear_cache()
        start = time.perf_counter()
        cached_encoder.encode(text)
        miss_time = (time.perf_counter() - start) * 1000

        # Second encode (hit)
        start = time.perf_counter()
        cached_encoder.encode(text)
        hit_time = (time.perf_counter() - start) * 1000

        print("\nCache performance:")
        print(f"  Miss: {miss_time:.4f}ms")
        print(f"  Hit: {hit_time:.4f}ms")
        print(f"  Speedup: {miss_time/hit_time:.1f}x")

        # Cache hit should be at least 10x faster
        assert hit_time < miss_time / 10, "Cache hit not significantly faster"

    def test_cache_info_updated(self, cached_encoder: YEncoder) -> None:
        """Cache statistics are correctly tracked."""
        cached_encoder.clear_cache()

        # Generate unique texts
        texts = [f"Unique text {i}" for i in range(5)]
        for text in texts:
            cached_encoder.encode(text)

        info = cached_encoder.cache_info()
        assert info is not None
        assert info.misses == 5
        assert info.currsize == 5

        # Re-encode same texts (hits)
        for text in texts:
            cached_encoder.encode(text)

        info = cached_encoder.cache_info()
        assert info is not None
        assert info.hits == 5


@pytest.mark.benchmark
class TestThroughput:
    """Benchmark tests for encoding throughput."""

    def test_encoding_throughput(self, encoder: YEncoder) -> None:
        """Measure texts per second throughput."""
        texts = [f"Short text {i}" for i in range(100)]

        start = time.perf_counter()
        for text in texts:
            encoder.encode(text)
        elapsed = time.perf_counter() - start

        throughput = len(texts) / elapsed

        print("\nThroughput (single encode):")
        print(f"  {throughput:.1f} texts/second")
        print(f"  {elapsed/len(texts)*1000:.2f}ms per text")

    def test_batch_throughput(self, encoder: YEncoder) -> None:
        """Measure batch encoding throughput."""
        texts = [f"Text for throughput test {i}" for i in range(100)]

        start = time.perf_counter()
        encoder.encode_batch(texts)
        elapsed = time.perf_counter() - start

        throughput = len(texts) / elapsed

        print("\nThroughput (batch encode):")
        print(f"  {throughput:.1f} texts/second")
        print(f"  {elapsed/len(texts)*1000:.2f}ms per text")
