"""Embedding generation using Sentence Transformers."""

from __future__ import annotations

import os
from typing import Sequence

from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class EmbeddingManager:
    """Wraps SentenceTransformer to produce normalized dense vector embeddings."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{self.model_name}': {exc}"
            ) from exc

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Generate normalized vector embeddings for a sequence of document chunks."""
        clean_texts = [t.strip() for t in texts if t and t.strip()]
        if not clean_texts:
            return []
        if len(clean_texts) != len(texts):
            raise ValueError("All document chunks passed for embedding must be non-empty.")

        vectors = self._model.encode(
            list(clean_texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized vector embedding for a single user query."""
        cleaned = (query or "").strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty.")

        vector = self._model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def get_model_info(self) -> dict[str, str | int]:
        """Return metadata about the loaded embedding model."""
        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
        else:
            dim = self._model.get_sentence_embedding_dimension()
        return {
            "model_name": self.model_name,
            "embedding_dimension": int(dim) if dim is not None else 0,
        }
