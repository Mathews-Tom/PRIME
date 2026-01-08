"""Latency benchmark tests for SSM (Semantic State Monitor).

Tests measure SSM latency targets:
- Variance calculation only: <5ms p50
- End-to-end with encoder (CPU): <200ms p50
- End-to-end with encoder (GPU): <50ms p50 (not validated in CI)

Run with: uv run pytest tests/benchmark/ssm/ -v -m benchmark
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from prime.encoder import MINILM_CONFIG, YEncoder
from prime.ssm import SSMConfig, SemanticStateMonitor
from prime.ssm.ssm_types import ActionState

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MockFastEncoder:
    """Fast mock encoder for isolating SSM variance calculation latency.

    Returns deterministic embeddings without model inference overhead.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self._embedding_dim = embedding_dim
        self._counter = 0

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def max_length(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-fast-encoder"

    def encode(self, text: str) -> NDArray[np.float32]:
        """Return deterministic normalized embedding."""
        # Create pseudo-random but deterministic embedding from hash
        seed = hash(text) % (2**32)
        rng = np.random.Generator(np.random.PCG64(seed))
        embedding = rng.normal(0, 1, self._embedding_dim).astype(np.float32)
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        self._counter += 1
        return embedding

    def encode_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [self.encode(t) for t in texts]

    def get_model_info(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "max_length": self.max_length,
            "is_mock": True,
        }


@pytest.fixture(scope="module")
def real_encoder() -> YEncoder:
    """Real MiniLM encoder for end-to-end benchmarks."""
    return YEncoder(MINILM_CONFIG)


@pytest.fixture(scope="module")
def mock_encoder() -> MockFastEncoder:
    """Fast mock encoder for variance-only benchmarks."""
    return MockFastEncoder(embedding_dim=384)


@pytest.fixture
def ssm_config() -> SSMConfig:
    """Standard SSM config for benchmarks."""
    return SSMConfig(
        embedding_dim=384,
        window_size=5,
        variance_threshold=0.15,
        smoothing_factor=0.3,
    )


@pytest.mark.benchmark
class TestVarianceCalculationLatency:
    """Benchmark tests for SSM variance calculation only.

    Target: <5ms p50 for variance calculation (excluding encoding).
    """

    def test_variance_calc_latency_with_mock(
        self, mock_encoder: MockFastEncoder, ssm_config: SSMConfig
    ) -> None:
        """Measure variance calculation latency using mock encoder."""
        ssm = SemanticStateMonitor(mock_encoder, ssm_config)

        # Fill window first
        for i in range(5):
            ssm.update(f"Warmup message {i}")

        # Measure update latency (variance calc + state machine)
        iterations = 100
        times: list[float] = []
        for i in range(iterations):
            start = time.perf_counter()
            ssm.update(f"Benchmark message {i}")
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_ms = float(np.mean(times))
        p50_ms = float(np.percentile(times, 50))
        p95_ms = float(np.percentile(times, 95))
        p99_ms = float(np.percentile(times, 99))

        print("\nSSM Variance Calculation Latency (mock encoder):")
        print(f"  Average: {avg_ms:.4f}ms")
        print(f"  p50: {p50_ms:.4f}ms")
        print(f"  p95: {p95_ms:.4f}ms")
        print(f"  p99: {p99_ms:.4f}ms")

        # Target: <5ms p50 for variance calculation
        assert p50_ms < 5.0, f"p50 latency {p50_ms:.4f}ms exceeds 5ms threshold"

    def test_variance_scales_linearly_with_window(
        self, mock_encoder: MockFastEncoder
    ) -> None:
        """Verify variance calculation scales reasonably with window size."""
        window_sizes = [3, 5, 10, 20]
        latencies: dict[int, float] = {}

        for ws in window_sizes:
            config = SSMConfig(embedding_dim=384, window_size=ws)
            ssm = SemanticStateMonitor(mock_encoder, config)

            # Fill window
            for i in range(ws):
                ssm.update(f"Fill message {i}")

            # Measure
            iterations = 50
            times: list[float] = []
            for i in range(iterations):
                start = time.perf_counter()
                ssm.update(f"Measure {i}")
                end = time.perf_counter()
                times.append((end - start) * 1000)

            latencies[ws] = float(np.median(times))

        print("\nVariance calc latency by window size:")
        for ws, lat in latencies.items():
            print(f"  window_size={ws}: {lat:.4f}ms")

        # Verify all are under threshold
        for ws, lat in latencies.items():
            assert lat < 5.0, f"Window size {ws}: {lat:.4f}ms exceeds 5ms threshold"


@pytest.mark.benchmark
class TestEndToEndLatency:
    """Benchmark tests for end-to-end SSM update with real encoder.

    Target: <200ms p50 for CPU, <50ms p50 for GPU.
    """

    def test_e2e_latency_with_real_encoder(
        self, real_encoder: YEncoder, ssm_config: SSMConfig
    ) -> None:
        """Measure end-to-end latency with real MiniLM encoder."""
        ssm = SemanticStateMonitor(real_encoder, ssm_config)

        # Warmup (includes model warm-up)
        for i in range(5):
            ssm.update(f"Warmup message {i}")

        # Measure
        iterations = 20
        times: list[float] = []
        for i in range(iterations):
            start = time.perf_counter()
            ssm.update(f"This is benchmark message number {i} for latency testing")
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_ms = float(np.mean(times))
        p50_ms = float(np.percentile(times, 50))
        p95_ms = float(np.percentile(times, 95))

        print("\nSSM End-to-End Latency (real encoder, CPU):")
        print(f"  Average: {avg_ms:.2f}ms")
        print(f"  p50: {p50_ms:.2f}ms")
        print(f"  p95: {p95_ms:.2f}ms")

        # Target: <200ms p50 for CPU (generous for CI)
        assert p50_ms < 200, f"p50 latency {p50_ms:.2f}ms exceeds 200ms threshold"

    def test_e2e_latency_breakdown(
        self, real_encoder: YEncoder, ssm_config: SSMConfig
    ) -> None:
        """Break down latency into encoding vs variance calculation."""
        ssm = SemanticStateMonitor(real_encoder, ssm_config)

        # Warmup
        for i in range(5):
            ssm.update(f"Warmup {i}")

        # Measure encoding separately
        encode_times: list[float] = []
        for i in range(20):
            start = time.perf_counter()
            real_encoder.encode(f"Encoding benchmark {i}")
            encode_times.append((time.perf_counter() - start) * 1000)

        # Measure full update
        update_times: list[float] = []
        for i in range(20):
            start = time.perf_counter()
            ssm.update(f"Update benchmark {i}")
            update_times.append((time.perf_counter() - start) * 1000)

        encode_p50 = float(np.percentile(encode_times, 50))
        update_p50 = float(np.percentile(update_times, 50))
        overhead_p50 = update_p50 - encode_p50

        print("\nLatency Breakdown:")
        print(f"  Encoding p50: {encode_p50:.2f}ms")
        print(f"  Full update p50: {update_p50:.2f}ms")
        print(f"  SSM overhead p50: {overhead_p50:.2f}ms")

        # SSM overhead (variance + state machine) should be small
        assert overhead_p50 < 5.0, f"SSM overhead {overhead_p50:.2f}ms exceeds 5ms"


@pytest.mark.benchmark
class TestThroughput:
    """Benchmark tests for SSM throughput."""

    def test_updates_per_second_mock(
        self, mock_encoder: MockFastEncoder, ssm_config: SSMConfig
    ) -> None:
        """Measure SSM updates per second with mock encoder."""
        ssm = SemanticStateMonitor(mock_encoder, ssm_config)

        # Warmup
        for i in range(5):
            ssm.update(f"Warmup {i}")

        num_updates = 1000
        start = time.perf_counter()
        for i in range(num_updates):
            ssm.update(f"Throughput test message {i}")
        elapsed = time.perf_counter() - start

        throughput = num_updates / elapsed

        print("\nSSM Throughput (mock encoder):")
        print(f"  {throughput:.1f} updates/second")
        print(f"  {elapsed/num_updates*1000:.4f}ms per update")

    def test_updates_per_second_real(
        self, real_encoder: YEncoder, ssm_config: SSMConfig
    ) -> None:
        """Measure SSM updates per second with real encoder."""
        ssm = SemanticStateMonitor(real_encoder, ssm_config)

        # Warmup
        for i in range(5):
            ssm.update(f"Warmup {i}")

        num_updates = 50  # Fewer due to encoding overhead
        start = time.perf_counter()
        for i in range(num_updates):
            ssm.update(f"Throughput test message {i}")
        elapsed = time.perf_counter() - start

        throughput = num_updates / elapsed

        print("\nSSM Throughput (real encoder):")
        print(f"  {throughput:.1f} updates/second")
        print(f"  {elapsed/num_updates*1000:.2f}ms per update")


@pytest.mark.benchmark
class TestActionStateTransitionPerformance:
    """Benchmark action state determination."""

    def test_state_determination_overhead(
        self, mock_encoder: MockFastEncoder
    ) -> None:
        """Measure overhead of state determination logic."""
        # Create SSM with very low threshold to trigger all states
        config = SSMConfig(
            embedding_dim=384,
            window_size=5,
            variance_threshold=0.001,  # Very low to trigger transitions
        )
        ssm = SemanticStateMonitor(mock_encoder, config)

        # Warmup
        for i in range(5):
            ssm.update(f"Warmup {i}")

        # Collect state distribution while measuring
        iterations = 100
        state_counts: dict[ActionState, int] = {s: 0 for s in ActionState}
        times: list[float] = []

        for i in range(iterations):
            start = time.perf_counter()
            result = ssm.update(f"State test {i}")
            times.append((time.perf_counter() - start) * 1000)
            state_counts[result.action] += 1

        p50_ms = float(np.percentile(times, 50))

        print("\nState Determination Performance:")
        print(f"  p50 latency: {p50_ms:.4f}ms")
        print("  State distribution:")
        for state, count in state_counts.items():
            print(f"    {state.value}: {count}")

        # State determination should add negligible overhead
        assert p50_ms < 5.0


@pytest.mark.benchmark
class TestMemoryEfficiency:
    """Benchmark memory usage patterns."""

    def test_window_buffer_memory(
        self, mock_encoder: MockFastEncoder
    ) -> None:
        """Verify window buffer maintains constant memory."""
        config = SSMConfig(embedding_dim=384, window_size=10)
        ssm = SemanticStateMonitor(mock_encoder, config)

        # Simulate long conversation
        for i in range(1000):
            ssm.update(f"Message {i} for memory test")

        state = ssm.get_state()
        assert state["window_size"] == 10, "Window should stay at max size"
        assert state["turn_number"] == 1000, "Turn counter should increment"

        print("\nMemory Efficiency:")
        print(f"  Turns processed: {state['turn_number']}")
        print(f"  Window size (capped): {state['window_size']}")
