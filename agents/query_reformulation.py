"""
Agent 1 — Query Reformulation Agent
Rewrites the user query into multiple targeted sub-queries to maximise
recall and reduce retrieval misses.
"""
from __future__ import annotations

import logging
import re
from typing import List

from langchain.prompts import PromptTemplate

from agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""You are a query reformulation specialist for a Retrieval-Augmented Generation (RAG) system.

Your task is to rewrite the following user query into 3 diverse, specific sub-queries that will 
help retrieve the most relevant information from a knowledge base.

Rules:
- Each sub-query should target a different aspect of the original question
- Be specific and concrete — avoid vague terms
- Use different keywords/synonyms in each sub-query
- Keep each sub-query concise (under 20 words)
- Output ONLY the 3 sub-queries, one per line, numbered 1. 2. 3.

Original query: {query}

Sub-queries:""",
)


class QueryReformulationAgent:
    """Decomposes a user query into multiple retrieval-optimized sub-queries."""

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.3)
        self._chain = _PROMPT | self._llm

    def run(self, query: str) -> List[str]:
        """
        Returns list of sub-queries (always includes the original).
        Falls back gracefully if LLM fails.
        """
        logger.info("[QueryReformulation] Reformulating: %r", query)
        try:
            result = self._chain.invoke({"query": query})
            raw = result if isinstance(result, str) else result.content
            sub_queries = self._parse(raw)
            logger.info("[QueryReformulation] Generated %d sub-queries", len(sub_queries))
            return sub_queries
        except Exception as exc:
            logger.warning("[QueryReformulation] Failed: %s — using original query", exc)
            return [query]

    @staticmethod
    def _parse(text: str) -> List[str]:
        """Extract numbered sub-queries from LLM output."""
        lines = text.strip().splitlines()
        queries: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove leading numbering like "1.", "1)", "- ", etc.
            cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            cleaned = re.sub(r"^[-*•]\s*", "", cleaned).strip()
            if cleaned:
                queries.append(cleaned)
        # Always keep at least one meaningful result
        return queries[:3] if queries else []
