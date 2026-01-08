#!/usr/bin/env python3
"""PRIME Library Quickstart Example.

This example demonstrates in-process usage of PRIME as a Python library.
No external services (Qdrant, APIs) are required - everything runs in-memory.

Run with:
    uv run python examples/library_quickstart.py
"""

from __future__ import annotations

from prime import PRIME, PRIMEConfig


def main() -> None:
    """Demonstrate PRIME library usage."""
    print("=" * 60)
    print("PRIME Library Quickstart")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Initialize PRIME with testing configuration
    # -------------------------------------------------------------------------
    print("\n1. Initializing PRIME with for_testing() config...")
    print("   (Uses in-memory FAISS index, small embedding model)")

    config = PRIMEConfig.for_testing()
    prime = PRIME(config)

    print(f"   Embedding dimension: {config.y_encoder.embedding_dim}")
    print(f"   SSM variance threshold: {config.ssm.variance_threshold}")
    print(f"   SSM window size: {config.ssm.window_size}")
    print("   PRIME initialized successfully!")

    # -------------------------------------------------------------------------
    # 2. Ingest some knowledge into memory
    # -------------------------------------------------------------------------
    print("\n2. Ingesting external knowledge...")

    # Ingest knowledge about JEPA
    result = prime.write_external_knowledge(
        content=(
            "JEPA (Joint Embedding Predictive Architecture) is a machine learning "
            "approach that predicts in embedding space rather than pixel or token space. "
            "This allows the model to focus on abstract, high-level features."
        ),
        metadata={"source": "ml_concepts", "topic": "jepa"},
    )
    print(f"   Written: jepa       -> cluster {result.cluster_id}")

    # Ingest knowledge about RAG
    result = prime.write_external_knowledge(
        content=(
            "Retrieval-Augmented Generation (RAG) combines retrieval systems with "
            "generative models. Traditional RAG searches for relevant documents on every "
            "query, which can be inefficient."
        ),
        metadata={"source": "ml_concepts", "topic": "rag"},
    )
    print(f"   Written: rag        -> cluster {result.cluster_id}")

    # Ingest knowledge about SSM
    result = prime.write_external_knowledge(
        content=(
            "PRIME uses a Semantic State Monitor (SSM) to track conversation "
            "trajectory. It calculates variance from a centroid embedding and only "
            "triggers retrieval when a semantic boundary is crossed."
        ),
        metadata={"source": "prime_docs", "topic": "ssm"},
    )
    print(f"   Written: ssm        -> cluster {result.cluster_id}")

    # Ingest knowledge about MCS
    result = prime.write_external_knowledge(
        content=(
            "The Memory Cluster Store (MCS) in PRIME automatically groups similar "
            "memories into clusters. When a cluster grows large enough, it consolidates "
            "into a prototype embedding for efficient retrieval."
        ),
        metadata={"source": "prime_docs", "topic": "mcs"},
    )
    print(f"   Written: mcs        -> cluster {result.cluster_id}")

    # Ingest knowledge about Python (different topic)
    result = prime.write_external_knowledge(
        content=(
            "Python is a high-level programming language known for its readability. "
            "It was created by Guido van Rossum and first released in 1991."
        ),
        metadata={"source": "general", "topic": "python"},
    )
    print(f"   Written: python     -> cluster {result.cluster_id}")

    # -------------------------------------------------------------------------
    # 3. Simulate a conversation with predictive retrieval
    # -------------------------------------------------------------------------
    print("\n3. Simulating conversation with predictive retrieval...")
    print("   (Watch how SSM decides when to retrieve)")

    session_id = "demo_session"
    conversation = [
        # Turn 1: Establish topic
        "What is JEPA and how does it work?",
        # Turn 2: Follow-up on same topic (low variance expected)
        "What are the advantages of predicting in embedding space?",
        # Turn 3: Topic shift (high variance expected)
        "How does Python handle memory management?",
        # Turn 4: Another shift back to ML
        "Tell me about retrieval augmented generation.",
        # Turn 5: Force retrieval to demonstrate override
        "What is the Memory Cluster Store?",
    ]

    for i, query in enumerate(conversation, 1):
        print(f"\n   --- Turn {i} ---")
        print(f"   Query: {query[:60]}...")

        # Use force_retrieval on the last turn to demonstrate the override
        force = i == 5

        response = prime.process_turn(
            query,
            session_id=session_id,
            force_retrieval=force,
            k=3,
        )

        print(f"   Action: {response.action.value}")
        print(f"   Variance: {response.variance:.4f} (threshold: {config.ssm.variance_threshold})")
        print(f"   Boundary crossed: {response.boundary_crossed}")

        if response.retrieved_memories:
            print(f"   Retrieved {len(response.retrieved_memories)} memories:")
            for mem in response.retrieved_memories[:2]:  # Show top 2
                preview = mem.content[:80].replace("\n", " ")
                print(f"      [{mem.similarity:.2f}] {preview}...")

        # Simulate recording an LLM response
        if i == 1:
            prime.record_response(
                "JEPA predicts representations in embedding space, focusing on semantic "
                "features rather than low-level details like pixels.",
                session_id=session_id,
            )
            print("   (Recorded response to memory)")

    # -------------------------------------------------------------------------
    # 4. Direct memory search (bypasses SSM)
    # -------------------------------------------------------------------------
    print("\n4. Direct memory search (bypasses SSM boundary detection)...")

    results = prime.search_memory(
        query="semantic boundary detection",
        k=3,
        min_similarity=0.1,
    )

    print(f"   Found {len(results)} results for 'semantic boundary detection':")
    for mem in results:
        preview = mem.content[:80].replace("\n", " ")
        print(f"   [{mem.similarity:.2f}] {preview}...")

    # -------------------------------------------------------------------------
    # 5. System diagnostics
    # -------------------------------------------------------------------------
    print("\n5. System diagnostics...")

    diagnostics = prime.get_diagnostics()
    print(f"   Status: {diagnostics.status}")
    print(f"   Version: {diagnostics.version}")
    print(f"   Uptime: {diagnostics.uptime_seconds:.1f}s")
    print(f"   Total requests: {int(diagnostics.metrics['total_requests'])}")
    print(f"   Memory count: {int(diagnostics.metrics['mcs_memory_count'])}")
    print(f"   Cluster count: {int(diagnostics.metrics['mcs_cluster_count'])}")

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)
    print("""
Next steps:
  - Try the REST API: examples/api_server.md
  - Use the Python SDK: examples/sdk_client.py
  - Read the full docs: docs/PRIME-Project-Overview.md
""")


if __name__ == "__main__":
    main()
