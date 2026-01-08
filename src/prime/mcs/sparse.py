"""BM25 sparse vector generation for hybrid search.

Provides BM25Tokenizer for computing sparse vectors with term frequency
and inverse document frequency (IDF) weighting for keyword search.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from prime.mcs.index import SparseVector


@dataclass
class BM25Tokenizer:
    """BM25 tokenizer for sparse vector generation.

    Converts text to sparse vectors suitable for BM25-style keyword search.
    Tracks vocabulary and computes IDF weights across documents.

    Attributes:
        min_token_length: Minimum token length to include.
        max_vocab_size: Maximum vocabulary size (hash space).
        vocab: Token to index mapping.
        doc_count: Number of documents processed.
        doc_freq: Document frequency per token.

    Example:
        >>> tokenizer = BM25Tokenizer()
        >>> tokenizer.fit(["user prefers dark mode", "user likes large fonts"])
        >>> sparse = tokenizer.encode("dark mode preference")
        >>> print(f"Indices: {sparse.indices}, Values: {sparse.values}")
    """

    min_token_length: int = 2
    max_vocab_size: int = 30000
    vocab: dict[str, int] = field(default_factory=dict)
    doc_count: int = 0
    doc_freq: dict[str, int] = field(default_factory=dict)

    # BM25 parameters
    k1: float = 1.2  # Term frequency saturation parameter
    b: float = 0.75  # Document length normalization parameter
    avg_doc_length: float = 0.0

    # Pre-compiled regex for tokenization
    _token_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"[a-zA-Z0-9]+"),
        repr=False,
    )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase filtered tokens.

        Applies lowercase normalization, extracts alphanumeric tokens,
        and filters by minimum length.

        Args:
            text: Input text to tokenize.

        Returns:
            List of filtered tokens.

        Example:
            >>> tokenizer = BM25Tokenizer(min_token_length=2)
            >>> tokenizer.tokenize("Hello World! I am here.")
            ['hello', 'world', 'am', 'here']
        """
        if not text:
            return []

        # Extract alphanumeric tokens, lowercase, filter by length
        tokens = self._token_pattern.findall(text.lower())
        return [t for t in tokens if len(t) >= self.min_token_length]

    def fit(self, documents: list[str]) -> None:
        """Fit tokenizer on document corpus.

        Builds vocabulary and computes IDF weights from documents.

        Args:
            documents: List of documents to fit on.

        Example:
            >>> tokenizer = BM25Tokenizer()
            >>> tokenizer.fit(["doc one", "doc two"])
            >>> print(tokenizer.doc_count)
            2
        """
        self.doc_count = len(documents)
        self.doc_freq.clear()
        self.vocab.clear()

        total_length = 0

        for doc in documents:
            tokens = self.tokenize(doc)
            total_length += len(tokens)

            # Count unique tokens per document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

                # Add to vocabulary if not present
                if token not in self.vocab:
                    # Use hash-based index for consistent mapping
                    idx = abs(hash(token)) % self.max_vocab_size
                    self.vocab[token] = idx

        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0.0

    def fit_incremental(self, document: str) -> None:
        """Incrementally update tokenizer with a single document.

        Updates vocabulary and IDF weights for online learning.

        Args:
            document: Document to add to corpus statistics.
        """
        tokens = self.tokenize(document)
        self.doc_count += 1

        # Update average document length
        if self.doc_count == 1:
            self.avg_doc_length = float(len(tokens))
        else:
            # Incremental mean update
            prev_total = self.avg_doc_length * (self.doc_count - 1)
            self.avg_doc_length = (prev_total + len(tokens)) / self.doc_count

        # Count unique tokens
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

            # Add to vocabulary if not present
            if token not in self.vocab:
                idx = abs(hash(token)) % self.max_vocab_size
                self.vocab[token] = idx

    def get_idf(self, token: str) -> float:
        """Compute IDF weight for a token.

        Uses smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        where N is document count and df is document frequency.

        Args:
            token: Token to compute IDF for.

        Returns:
            IDF weight (default 1.0 for unknown tokens).
        """
        if self.doc_count == 0:
            return 1.0

        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 1.0

        # BM25 IDF formula with smoothing
        idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        return max(idf, 0.0)  # Ensure non-negative

    def encode(self, text: str) -> SparseVector:
        """Encode text to sparse vector with BM25 weighting.

        Computes TF-IDF style weights using BM25 formula:
        score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))

        Args:
            text: Input text to encode.

        Returns:
            SparseVector with vocabulary indices and BM25 weights.

        Example:
            >>> tokenizer = BM25Tokenizer()
            >>> tokenizer.fit(["the quick brown fox", "lazy dog"])
            >>> sparse = tokenizer.encode("quick fox")
            >>> len(sparse.indices) > 0
            True
        """
        tokens = self.tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Compute term frequencies
        tf_counts = Counter(tokens)
        doc_length = len(tokens)

        indices: list[int] = []
        values: list[float] = []

        for token, tf in tf_counts.items():
            # Get or create vocabulary index
            if token in self.vocab:
                idx = self.vocab[token]
            else:
                idx = abs(hash(token)) % self.max_vocab_size

            # Compute BM25 score
            idf = self.get_idf(token)

            # BM25 term frequency component
            if self.avg_doc_length > 0:
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                )
            else:
                tf_component = float(tf)

            score = idf * tf_component

            if score > 0:
                indices.append(idx)
                values.append(score)

        # Sort by index for consistent ordering
        if indices:
            sorted_pairs = sorted(zip(indices, values, strict=True))
            indices = [p[0] for p in sorted_pairs]
            values = [p[1] for p in sorted_pairs]

        return SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVector:
        """Encode query text to sparse vector.

        Uses simplified weighting for queries (no length normalization).

        Args:
            text: Query text to encode.

        Returns:
            SparseVector optimized for query matching.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Compute term frequencies
        tf_counts = Counter(tokens)

        indices: list[int] = []
        values: list[float] = []

        for token, tf in tf_counts.items():
            # Get or create vocabulary index
            if token in self.vocab:
                idx = self.vocab[token]
            else:
                idx = abs(hash(token)) % self.max_vocab_size

            # Query weighting: IDF * (k1 + 1) * tf / (k1 + tf)
            idf = self.get_idf(token)
            score = idf * (self.k1 + 1) * tf / (self.k1 + tf)

            if score > 0:
                indices.append(idx)
                values.append(score)

        # Sort by index for consistent ordering
        if indices:
            sorted_pairs = sorted(zip(indices, values, strict=True))
            indices = [p[0] for p in sorted_pairs]
            values = [p[1] for p in sorted_pairs]

        return SparseVector(indices=indices, values=values)


def rrf_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine dense and sparse results using Reciprocal Rank Fusion.

    RRF formula: score = sum(1 / (k + rank)) across all result lists.

    Args:
        dense_results: List of (id, score) from dense search.
        sparse_results: List of (id, score) from sparse search.
        k: RRF parameter (default 60, higher = more emphasis on top ranks).

    Returns:
        List of (id, fused_score) sorted by score descending.

    Example:
        >>> dense = [("doc1", 0.9), ("doc2", 0.8)]
        >>> sparse = [("doc2", 5.0), ("doc3", 3.0)]
        >>> fused = rrf_fusion(dense, sparse)
        >>> fused[0][0]  # Top result ID
        'doc2'
    """
    scores: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # Sort by fused score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
