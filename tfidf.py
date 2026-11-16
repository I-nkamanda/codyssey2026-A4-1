"""NumPy-based TF-IDF vectorizer and cosine similarity calculation."""
from __future__ import annotations

from collections import Counter

import numpy as np

from preprocess import tokenize


class Tfidf:
    """TF-IDF vectorizer using raw TF, smoothed IDF, and L2 normalization."""

    def fit(self, documents: list[str]) -> 'Tfidf':
        """Build vocabulary and compute smoothed IDF from training documents."""
        tokens = [tokenize(d) for d in documents]
        self.vocabulary = {word: i for i, word in enumerate(sorted({w for doc in tokens for w in doc}))}
        counts = self._counts(tokens)
        self.idf = np.log((1 + len(documents)) / (1 + np.count_nonzero(counts, axis=0))) + 1
        return self

    def _counts(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        """Build a (num_docs, vocab_size) term-frequency matrix."""
        matrix = np.zeros((len(tokenized_docs), len(self.vocabulary)), dtype=np.float64)
        for i, words in enumerate(tokenized_docs):
            for word, count in Counter(words).items():
                if word in self.vocabulary:
                    matrix[i, self.vocabulary[word]] = count
        return matrix

    def transform(self, documents: list[str]) -> np.ndarray:
        """Transform documents into L2-normalized TF-IDF vectors."""
        matrix = self._counts([tokenize(d) for d in documents]) * self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and each row of matrix."""
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(matrix @ query, denominator, out=np.zeros(len(matrix)), where=denominator != 0)
