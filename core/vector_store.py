"""
Vector store module — wraps ChromaDB with a clean interface.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from core.config import config
from core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed persistent vector store."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._emb = get_embedding_model()
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector store ready — collection '%s' has %d documents",
            config.CHROMA_COLLECTION,
            self._collection.count(),
        )

    # ── Ingestion ──────────────────────────────────────────────
    def add_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        Add a list of document dicts.
        Each dict must have: {"text": str, "metadata": dict}
        Returns number of documents added.
        """
        if not docs:
            return 0

        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        ids = [str(uuid.uuid4()) for _ in docs]
        embeddings = self._emb.embed(texts)

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d documents to vector store", len(docs))
        return len(docs)

    # ── Retrieval ──────────────────────────────────────────────
    def similarity_search(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k chunks most similar to `query`.
        Each result: {"text": str, "metadata": dict, "score": float}
        """
        k = k or config.TOP_K_CHUNKS
        q_vec = self._emb.embed_single(query)

        results = self._collection.query(
            query_embeddings=[q_vec],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance → similarity score in [0, 1]
            score = 1.0 - float(dist)
            chunks.append({"text": text, "metadata": meta, "score": score})

        return chunks

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents (use with caution)."""
        self._client.delete_collection(config.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store collection reset.")


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
