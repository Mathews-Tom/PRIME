#!/usr/bin/env python3
"""PRIME SDK Client Example.

This example demonstrates using the PRIME Python SDK to interact with
the PRIME REST API. Both async and sync clients are shown.

Prerequisites:
    1. Start the PRIME server:
       uv run uvicorn prime.api.app:app --host 0.0.0.0 --port 8000

    2. Run this example:
       uv run python examples/sdk_client.py
"""

from __future__ import annotations

import asyncio

from prime.client import PRIMEClient, PRIMEClientSync


def run_sync_example() -> None:
    """Demonstrate synchronous SDK usage."""
    print("=" * 60)
    print("PRIME SDK - Synchronous Client")
    print("=" * 60)

    with PRIMEClientSync(base_url="http://localhost:8000") as client:
        # Health check
        print("\n1. Health check...")
        health = client.health()
        print(f"   Status: {health.status}")
        print(f"   Version: {health.version}")

        # Write some memories
        print("\n2. Writing memories...")

        memories = [
            "The Transformer architecture uses self-attention mechanisms.",
            "BERT is a bidirectional encoder representation from transformers.",
            "GPT models are autoregressive language models.",
        ]

        for content in memories:
            result = client.write_memory(
                content=content,
                metadata={"source": "ml_knowledge"},
            )
            print(f"   Written: {content[:50]}... -> cluster {result.cluster_id}")

        # Process a turn (uses SSM boundary detection)
        print("\n3. Processing turns with boundary detection...")

        turns = [
            "What is the Transformer architecture?",
            "How does self-attention work?",  # Same topic - might not trigger retrieval
            "What is Python used for?",  # Topic shift - should trigger retrieval
        ]

        session_id = "sdk_demo_session"

        for query in turns:
            response = client.process_turn(
                content=query,
                session_id=session_id,
                k=3,
            )

            print(f"\n   Query: {query}")
            print(f"   Action: {response.action}")
            print(f"   Boundary crossed: {response.boundary_crossed}")
            print(f"   Variance: {response.variance:.4f}")

            if response.retrieved_memories:
                print(f"   Retrieved {len(response.retrieved_memories)} memories:")
                for mem in response.retrieved_memories[:2]:
                    print(f"      [{mem.similarity:.2f}] {mem.content[:60]}...")

        # Direct search (bypasses SSM)
        print("\n4. Direct memory search...")
        search_result = client.search(
            query="transformer attention",
            k=3,
            min_similarity=0.1,
        )

        print(f"   Found {len(search_result.results)} results:")
        for mem in search_result.results:
            print(f"   [{mem.similarity:.2f}] {mem.content[:60]}...")

        # Get diagnostics
        print("\n5. System diagnostics...")
        diagnostics = client.diagnostics()
        print(f"   Status: {diagnostics.get('status', 'unknown')}")
        print(f"   Uptime: {diagnostics.get('uptime_seconds', 0):.1f}s")

    print("\n" + "=" * 60)
    print("Sync example complete!")


async def run_async_example() -> None:
    """Demonstrate asynchronous SDK usage."""
    print("\n" + "=" * 60)
    print("PRIME SDK - Asynchronous Client")
    print("=" * 60)

    async with PRIMEClient(base_url="http://localhost:8000") as client:
        # Health check
        print("\n1. Async health check...")
        health = await client.health()
        print(f"   Status: {health.status}")
        print(f"   Version: {health.version}")

        # Parallel memory writes (async advantage)
        print("\n2. Parallel memory writes...")

        contents = [
            "Neural networks are inspired by biological neurons.",
            "Backpropagation is used to train neural networks.",
            "Gradient descent optimizes the loss function.",
        ]

        # Write all memories in parallel
        tasks = [
            client.write_memory(
                content=c,
                metadata={"source": "ml_knowledge", "async": "true"},
            )
            for c in contents
        ]
        results = await asyncio.gather(*tasks)

        for content, result in zip(contents, results):
            print(f"   Written: {content[:40]}... -> cluster {result.cluster_id}")

        # Process a turn
        print("\n3. Async turn processing...")
        response = await client.process_turn(
            content="How do neural networks learn?",
            session_id="async_demo_session",
            force_retrieval=True,  # Force retrieval to demonstrate
            k=5,
        )

        print(f"   Action: {response.action}")
        print(f"   Variance: {response.variance:.4f}")
        print(f"   Retrieved {len(response.retrieved_memories)} memories")

        # Async search
        print("\n4. Async search...")
        search_result = await client.search(
            query="neural network training",
            k=3,
        )

        print(f"   Found {len(search_result.results)} results")

    print("\n" + "=" * 60)
    print("Async example complete!")


def main() -> None:
    """Run both sync and async examples."""
    print("""
NOTE: This example requires the PRIME server to be running.
      Start it with: uv run uvicorn prime.api.app:app --port 8000

      If you get connection errors, make sure the server is up!
""")

    try:
        # Run synchronous example
        run_sync_example()

        # Run asynchronous example
        asyncio.run(run_async_example())

        print("\n" + "=" * 60)
        print("All SDK examples complete!")
        print("=" * 60)

    except Exception as e:
        if "Connection refused" in str(e) or "ConnectError" in str(e):
            print(f"\nERROR: Could not connect to PRIME server at localhost:8000")
            print("Make sure to start the server first:")
            print("    uv run uvicorn prime.api.app:app --port 8000")
        else:
            raise


if __name__ == "__main__":
    main()
