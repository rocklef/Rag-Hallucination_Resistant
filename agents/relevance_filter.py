"""
Agent 3 — Relevance Filter Agent
Scores each retrieved chunk's relevance to the user query using an LLM
and discards low-scoring chunks (below threshold).
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate

from agents.llm_factory import get_llm
from core.config import config

logger = logging.getLogger(__name__)

_PROMPT = PromptTemplate(
    input_variables=["query", "chunk"],
    template="""You are a relevance scoring expert. Rate how relevant the following text chunk is 
to answering the user's query.

User Query: {query}

Text Chunk:
\"\"\"
{chunk}
\"\"\"

Score this chunk on a scale from 0 to 10:
- 0-3: Irrelevant or off-topic
- 4-5: Marginally related
- 6-7: Relevant, partially answers the query
- 8-10: Highly relevant, directly answers the query

Respond with ONLY a number between 0 and 10. Nothing else.""",
)


class RelevanceFilterAgent:
    """LLM-based relevance scoring and filtering of retrieved chunks."""

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.0)
        self._chain = _PROMPT | self._llm

    def run(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Score each chunk and return only those above the threshold.

        Args:
            query: Original user query
            chunks: List of {text, metadata, score} from RetrieverAgent
            threshold: Minimum relevance score (0-10) to keep a chunk

        Returns:
            Filtered and annotated chunks with 'relevance_score' field
        """
        threshold = threshold if threshold is not None else config.RELEVANCE_THRESHOLD
        filtered: List[Dict[str, Any]] = []

        for chunk in chunks:
            score = self._score_chunk(query, chunk["text"])
            chunk = {**chunk, "relevance_score": score}
            logger.debug(
                "[RelevanceFilter] Score %.1f for: %s...",
                score,
                chunk["text"][:60],
            )
            if score >= threshold:
                filtered.append(chunk)

        # Sort by relevance score descending
        filtered.sort(key=lambda c: c["relevance_score"], reverse=True)

        logger.info(
            "[RelevanceFilter] Kept %d/%d chunks (threshold=%.1f)",
            len(filtered),
            len(chunks),
            threshold,
        )

        # Safety: if nothing passes filter, return top-3 by vector score
        if not filtered and chunks:
            logger.warning("[RelevanceFilter] No chunks passed filter — using top-3 by vector score")
            return sorted(chunks, key=lambda c: c["score"], reverse=True)[:3]

        return filtered

    def _score_chunk(self, query: str, chunk_text: str) -> float:
        """Score a single chunk using LLM. Returns float in [0, 10]."""
        try:
            # Truncate chunk to avoid token limits
            truncated = chunk_text[:1200]
            result = self._chain.invoke({"query": query, "chunk": truncated})
            raw = result if isinstance(result, str) else result.content
            return self._parse_score(raw.strip())
        except Exception as exc:
            logger.warning("[RelevanceFilter] Scoring failed: %s — defaulting to 5.0", exc)
            return 5.0

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract numeric score from LLM response."""
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        return 5.0  # default if parsing fails
