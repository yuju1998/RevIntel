"""Text embedding providers for CRM semantic search.

`SentenceTransformerProvider` loads the local model once per process, lazily and behind a
lock, and reuses it for every document batch and query. Every vector is checked to have
exactly `dimension` finite values and is L2-normalized, so pgvector's cosine distance
(`<=>`) ranks by cosine similarity.
"""

import math
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import structlog

logger = structlog.get_logger("app.embeddings")


class EmbeddingError(RuntimeError):
    """The embedding model cannot be loaded or produced invalid vectors."""


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int:
        """Most tokens one input may have before the model would truncate it."""
        ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def check_vectors(vectors: Any, expected_rows: int, dimension: int) -> list[list[float]]:
    """Plain float lists, after checking shape, dimension and finiteness."""
    rows = [[float(value) for value in vector] for vector in vectors]
    if len(rows) != expected_rows:
        raise EmbeddingError(f"expected {expected_rows} embeddings, got {len(rows)}")
    for row in rows:
        if len(row) != dimension:
            raise EmbeddingError(f"expected {dimension}-dimensional embeddings, got {len(row)}")
        if not all(math.isfinite(value) for value in row):
            raise EmbeddingError("embedding contains non-finite values")
    return rows


def _load_sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


class SentenceTransformerProvider:
    def __init__(
        self,
        model_name: str,
        dimension: int,
        *,
        batch_size: int = 64,
        loader: Callable[[str], Any] = _load_sentence_transformer,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._loader = loader
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get(self) -> Any:
        """The loaded model; loaded once on first use and checked. Failures are not cached."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    model = self._loader(self._model_name)
                except Exception as exc:
                    raise EmbeddingError(f"cannot load {self._model_name}: {exc}") from exc
                actual = model.get_embedding_dimension()
                if actual != self._dimension:
                    raise EmbeddingError(
                        f"{self._model_name} produces {actual}-dimensional embeddings; "
                        f"{self._dimension} are required"
                    )
                logger.info("embedding_model_loaded", model=self._model_name, dimension=actual)
                self._model = model
        return self._model

    @property
    def max_tokens(self) -> int:
        # The model's sequence limit includes the [CLS] and [SEP] special tokens.
        return int(self._get().max_seq_length) - 2

    def count_tokens(self, text: str) -> int:
        tokenizer = self._get().tokenizer
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return check_vectors(vectors, len(texts), self._dimension)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
