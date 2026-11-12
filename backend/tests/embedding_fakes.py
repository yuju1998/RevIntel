"""Deterministic embedding provider for tests (no model download, no torch).

Vectors are normalized bags of hashed words: texts sharing words are similar, identical
texts are identical. One token per whitespace-separated word.
"""

import hashlib
import math
import re
from collections.abc import Sequence

DIMENSION = 384
_WORD = re.compile(r"[a-z0-9]+")


def bag_of_words(text: str, dimension: int = DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    for word in _WORD.findall(text.lower()):
        vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dimension] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeEmbeddingProvider:
    def __init__(self, *, model_name: str = "fake-bag-of-words", max_tokens: int = 254) -> None:
        self.model_name = model_name
        self.dimension = DIMENSION
        self.max_tokens = max_tokens
        self.document_calls: list[list[str]] = []
        self.queries: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [bag_of_words(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return bag_of_words(text)
