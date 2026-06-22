"""
Embeddings module — wraps SentenceTransformers for local, free embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from core.config import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Singleton wrapper around SentenceTransformer."""

    _instance: "EmbeddingModel | None" = None

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return list of float vectors."""
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel.get()
