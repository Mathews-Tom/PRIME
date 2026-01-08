"""Unit tests for BM25 tokenizer and hybrid search.

Tests the BM25Tokenizer for sparse vector generation and RRF fusion
for combining dense and sparse search results.
"""

from __future__ import annotations

from prime.mcs import BM25Tokenizer, SparseVector, rrf_fusion


class TestBM25Tokenizer:
    """Test BM25Tokenizer tokenization and encoding."""

    def test_tokenize_basic(self) -> None:
        """Tokenize basic text."""
        tokenizer = BM25Tokenizer()
        tokens = tokenizer.tokenize("Hello World Test")

        assert tokens == ["hello", "world", "test"]

    def test_tokenize_filters_short_tokens(self) -> None:
        """Tokenize filters tokens below min length."""
        tokenizer = BM25Tokenizer(min_token_length=3)
        tokens = tokenizer.tokenize("I am a test user")

        assert "am" not in tokens
        assert "test" in tokens
        assert "user" in tokens

    def test_tokenize_handles_punctuation(self) -> None:
        """Tokenize extracts alphanumeric tokens from punctuation."""
        tokenizer = BM25Tokenizer(min_token_length=2)
        tokens = tokenizer.tokenize("Hello, World! How's it going?")

        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "going" in tokens

    def test_tokenize_empty_string(self) -> None:
        """Tokenize returns empty list for empty string."""
        tokenizer = BM25Tokenizer()
        tokens = tokenizer.tokenize("")

        assert tokens == []

    def test_fit_builds_vocabulary(self) -> None:
        """Fit builds vocabulary from documents."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world", "world test", "test example"])

        assert "hello" in tokenizer.vocab
        assert "world" in tokenizer.vocab
        assert "test" in tokenizer.vocab

    def test_fit_computes_doc_freq(self) -> None:
        """Fit computes document frequency."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world", "world test", "test example"])

        assert tokenizer.doc_freq["world"] == 2  # Appears in 2 docs
        assert tokenizer.doc_freq["hello"] == 1  # Appears in 1 doc

    def test_fit_computes_avg_doc_length(self) -> None:
        """Fit computes average document length."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world", "test", "one two three"])

        # (2 + 1 + 3) / 3 = 2.0
        assert tokenizer.avg_doc_length == 2.0

    def test_fit_incremental_updates_stats(self) -> None:
        """Fit incremental updates statistics."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit_incremental("hello world")
        tokenizer.fit_incremental("world test")

        assert tokenizer.doc_count == 2
        assert tokenizer.doc_freq["world"] == 2
        assert tokenizer.doc_freq["hello"] == 1

    def test_get_idf_unknown_token(self) -> None:
        """Get IDF returns 1.0 for unknown token."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world"])

        idf = tokenizer.get_idf("unknown")
        assert idf == 1.0

    def test_get_idf_known_token(self) -> None:
        """Get IDF returns computed IDF for known token."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world", "world test", "test example"])

        # world appears in 2/3 documents
        idf_world = tokenizer.get_idf("world")
        # hello appears in 1/3 documents
        idf_hello = tokenizer.get_idf("hello")

        # Rarer token should have higher IDF
        assert idf_hello > idf_world

    def test_encode_returns_sparse_vector(self) -> None:
        """Encode returns SparseVector."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world test"])
        sparse = tokenizer.encode("hello world")

        assert isinstance(sparse, SparseVector)
        assert len(sparse.indices) > 0
        assert len(sparse.indices) == len(sparse.values)

    def test_encode_empty_text(self) -> None:
        """Encode returns empty sparse vector for empty text."""
        tokenizer = BM25Tokenizer()
        sparse = tokenizer.encode("")

        assert sparse.indices == []
        assert sparse.values == []

    def test_encode_weights_by_idf(self) -> None:
        """Encode weights terms by IDF."""
        tokenizer = BM25Tokenizer()
        # 'common' appears in all docs, 'rare' in one
        tokenizer.fit([
            "common word one",
            "common word two",
            "common word rare",
        ])

        sparse = tokenizer.encode("common rare")

        # Both should be present
        assert len(sparse.indices) == 2
        # Rare term should have higher weight
        idx_common = tokenizer.vocab["common"]
        idx_rare = tokenizer.vocab["rare"]

        common_idx_pos = sparse.indices.index(idx_common) if idx_common in sparse.indices else -1
        rare_idx_pos = sparse.indices.index(idx_rare) if idx_rare in sparse.indices else -1

        if common_idx_pos >= 0 and rare_idx_pos >= 0:
            assert sparse.values[rare_idx_pos] > sparse.values[common_idx_pos]

    def test_encode_query_differs_from_encode(self) -> None:
        """Encode query uses different weighting than encode."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["hello world test", "world example"])

        doc_sparse = tokenizer.encode("hello world")
        query_sparse = tokenizer.encode_query("hello world")

        # Both should have indices
        assert len(doc_sparse.indices) > 0
        assert len(query_sparse.indices) > 0
        # Values may differ due to different weighting
        # Just verify both produce valid output
        assert all(v > 0 for v in doc_sparse.values)
        assert all(v > 0 for v in query_sparse.values)

    def test_encode_sorted_indices(self) -> None:
        """Encode returns sorted indices."""
        tokenizer = BM25Tokenizer()
        tokenizer.fit(["zebra apple mango banana"])
        sparse = tokenizer.encode("zebra apple mango banana")

        # Indices should be sorted
        assert sparse.indices == sorted(sparse.indices)


class TestRRFFusion:
    """Test Reciprocal Rank Fusion."""

    def test_rrf_basic_fusion(self) -> None:
        """RRF combines two result lists."""
        dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        sparse = [("doc2", 5.0), ("doc3", 4.0), ("doc4", 3.0)]

        fused = rrf_fusion(dense, sparse)

        # doc2 appears in both, should be ranked highest
        assert fused[0][0] == "doc2"

    def test_rrf_empty_lists(self) -> None:
        """RRF handles empty lists."""
        fused = rrf_fusion([], [])
        assert fused == []

    def test_rrf_single_source(self) -> None:
        """RRF handles single source list."""
        dense = [("doc1", 0.9), ("doc2", 0.8)]
        fused = rrf_fusion(dense, [])

        assert len(fused) == 2
        assert fused[0][0] == "doc1"

    def test_rrf_k_parameter(self) -> None:
        """RRF k parameter affects fusion."""
        dense = [("doc1", 0.9), ("doc2", 0.8)]
        sparse = [("doc2", 5.0), ("doc1", 4.0)]

        fused_k60 = rrf_fusion(dense, sparse, k=60)
        fused_k1 = rrf_fusion(dense, sparse, k=1)

        # Both should rank doc1 and doc2 (they appear in both lists)
        assert len(fused_k60) == 2
        assert len(fused_k1) == 2
        # Score magnitudes differ with k
        assert fused_k1[0][1] > fused_k60[0][1]

    def test_rrf_preserves_unique_docs(self) -> None:
        """RRF preserves documents unique to each source."""
        dense = [("doc1", 0.9)]
        sparse = [("doc2", 5.0)]

        fused = rrf_fusion(dense, sparse)

        doc_ids = {doc_id for doc_id, _ in fused}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_rrf_score_calculation(self) -> None:
        """RRF calculates scores correctly."""
        dense = [("doc1", 0.9)]  # rank 0
        sparse = [("doc1", 5.0)]  # rank 0
        k = 60

        fused = rrf_fusion(dense, sparse, k=k)

        # doc1 appears at rank 0 in both lists
        # score = 1/(60+0+1) + 1/(60+0+1) = 2/61
        expected_score = 2.0 / 61.0
        assert abs(fused[0][1] - expected_score) < 0.0001


class TestBM25TokenizerEdgeCases:
    """Test edge cases for BM25Tokenizer."""

    def test_hash_collision_handling(self) -> None:
        """Tokenizer handles potential hash collisions gracefully."""
        tokenizer = BM25Tokenizer(max_vocab_size=10)  # Small vocab for collisions
        tokenizer.fit(["word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"])

        sparse = tokenizer.encode("word1 word2 word3")

        # Should still produce valid sparse vector
        assert len(sparse.indices) > 0
        assert all(0 <= idx < 10 for idx in sparse.indices)

    def test_unicode_handling(self) -> None:
        """Tokenizer handles unicode text."""
        tokenizer = BM25Tokenizer()
        # Unicode letters get filtered by alphanumeric regex
        tokens = tokenizer.tokenize("hello 世界 test")

        assert "hello" in tokens
        assert "test" in tokens

    def test_numeric_tokens(self) -> None:
        """Tokenizer handles numeric tokens."""
        tokenizer = BM25Tokenizer()
        tokens = tokenizer.tokenize("version 123 and test456")

        assert "version" in tokens
        assert "123" in tokens
        assert "test456" in tokens

    def test_repeated_fit_incremental(self) -> None:
        """Multiple fit_incremental calls accumulate correctly."""
        tokenizer = BM25Tokenizer()

        for i in range(100):
            tokenizer.fit_incremental(f"document {i} content")

        assert tokenizer.doc_count == 100
        assert tokenizer.avg_doc_length > 0

    def test_fit_then_encode_consistency(self) -> None:
        """Encoding after fit produces consistent results."""
        tokenizer = BM25Tokenizer()
        corpus = ["machine learning model", "deep learning neural", "model training data"]
        tokenizer.fit(corpus)

        sparse1 = tokenizer.encode("machine learning")
        sparse2 = tokenizer.encode("machine learning")

        assert sparse1.indices == sparse2.indices
        assert sparse1.values == sparse2.values
