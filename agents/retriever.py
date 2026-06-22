"""
Agent 2 — Retriever Agent
Executes multi-query retrieval against the vector store and
deduplicates results.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

from core.config import config
from core.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Runs multiple sub-queries against the vector store, deduplicates, and returns ranked chunks."""

    def __init__(self) -> None:
        self._store = get_vector_store()

    def run(
        self,
        queries: List[str],
        k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and deduplicate chunks for all sub-queries.

        Args:
            queries: List of sub-queries (from QueryReformulationAgent)
            k: Number of top chunks per query

        Returns:
            Sorted, deduplicated list of {text, metadata, score}
        """
        k = k or config.TOP_K_CHUNKS
        seen_texts: set[str] = set()
        all_chunks: List[Dict[str, Any]] = []

        for query in queries:
            logger.info("[Retriever] Searching for: %r", query)
            try:
                results = self._store.similarity_search(query, k=k)
                for chunk in results:
                    # Deduplicate by content (first 200 chars as fingerprint)
                    fingerprint = chunk["text"][:200].strip()
                    if fingerprint not in seen_texts:
                        seen_texts.add(fingerprint)
                        all_chunks.append(chunk)
            except Exception as exc:
                logger.error("[Retriever] Error for query %r: %s", query, exc)

        # Sort by relevance score descending
        all_chunks.sort(key=lambda c: c["score"], reverse=True)

        logger.info(
            "[Retriever] Retrieved %d unique chunks from %d queries",
            len(all_chunks),
            len(queries),
        )
        return all_chunks
