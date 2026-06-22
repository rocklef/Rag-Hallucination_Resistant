"""
Agent 4 — Cross-Reference Agent
Compares retrieved chunks for factual consistency.
Detects conflicting claims across sources and flags them.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

from langchain.prompts import PromptTemplate

from agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

_CONFLICT_PROMPT = PromptTemplate(
    input_variables=["query", "chunks_text"],
    template="""You are a fact-checking specialist. Your task is to analyze the following retrieved 
text chunks and identify any factual conflicts or inconsistencies between them.

User Query: {query}

Retrieved Chunks:
{chunks_text}

Instructions:
1. Read all chunks carefully
2. Identify any contradictory facts, conflicting numbers, or inconsistent claims
3. If conflicts exist, describe them briefly
4. Assign a consistency score: HIGH (fully consistent), MEDIUM (minor differences), LOW (major conflicts)

Respond in this exact format:
CONSISTENCY: <HIGH|MEDIUM|LOW>
CONFLICTS: <description of conflicts, or "None detected">
RELIABLE_CHUNKS: <comma-separated indices of the most reliable/consistent chunks, e.g. "1,3,5">""",
)

_SYNTHESIS_PROMPT = PromptTemplate(
    input_variables=["query", "chunks_text"],
    template="""Given the following verified text chunks, extract the key facts that are 
consistently supported across multiple sources.

User Query: {query}

Chunks:
{chunks_text}

List the key verified facts (5-10 bullet points) that are well-supported by the sources:""",
)


class CrossReferenceAgent:
    """
    Validates retrieved chunks for factual consistency.
    Returns verified chunks + a consistency report.
    """

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.0)
        self._conflict_chain = _CONFLICT_PROMPT | self._llm
        self._synthesis_chain = _SYNTHESIS_PROMPT | self._llm

    def run(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Cross-reference chunks and return (verified_chunks, report).

        report = {
            "consistency": "HIGH"|"MEDIUM"|"LOW",
            "conflicts": str,
            "key_facts": str,
        }
        """
        if not chunks:
            return chunks, {"consistency": "LOW", "conflicts": "No chunks to check", "key_facts": ""}

        # Build formatted chunk text
        chunks_text = self._format_chunks(chunks)

        # Step 1: Detect conflicts
        logger.info("[CrossReference] Checking %d chunks for consistency", len(chunks))
        consistency, conflicts, reliable_indices = self._detect_conflicts(query, chunks_text)

        # Step 2: Filter to reliable chunks if consistency is LOW
        verified_chunks = chunks
        if consistency == "LOW" and reliable_indices:
            try:
                verified_chunks = [chunks[i] for i in reliable_indices if i < len(chunks)]
                if not verified_chunks:
                    verified_chunks = chunks
            except Exception:
                verified_chunks = chunks

        # Step 3: Extract key verified facts
        key_facts = self._extract_key_facts(query, self._format_chunks(verified_chunks))

        report = {
            "consistency": consistency,
            "conflicts": conflicts,
            "key_facts": key_facts,
        }

        logger.info(
            "[CrossReference] Consistency=%s | %d verified chunks",
            consistency,
            len(verified_chunks),
        )
        return verified_chunks, report

    def _detect_conflicts(
        self, query: str, chunks_text: str
    ) -> Tuple[str, str, List[int]]:
        """Run conflict detection LLM call and parse results."""
        try:
            result = self._conflict_chain.invoke({"query": query, "chunks_text": chunks_text})
            raw = result if isinstance(result, str) else result.content
            return self._parse_conflict_response(raw)
        except Exception as exc:
            logger.warning("[CrossReference] Conflict detection failed: %s", exc)
            return "MEDIUM", "Unable to detect conflicts", []

    def _extract_key_facts(self, query: str, chunks_text: str) -> str:
        """Extract key verified facts using LLM."""
        try:
            result = self._synthesis_chain.invoke({"query": query, "chunks_text": chunks_text})
            return result if isinstance(result, str) else result.content
        except Exception as exc:
            logger.warning("[CrossReference] Key fact extraction failed: %s", exc)
            return ""

    @staticmethod
    def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for LLM prompt."""
        parts = []
        for i, chunk in enumerate(chunks[:10], 1):  # Limit to 10 chunks
            source = chunk.get("metadata", {}).get("source", "unknown")
            parts.append(f"[Chunk {i}] (Source: {source})\n{chunk['text'][:600]}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_conflict_response(text: str) -> Tuple[str, str, List[int]]:
        """Parse the structured LLM conflict report."""
        consistency = "MEDIUM"
        conflicts = "Unable to parse"
        reliable_indices: List[int] = []

        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("CONSISTENCY:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in {"HIGH", "MEDIUM", "LOW"}:
                    consistency = val
            elif line.startswith("CONFLICTS:"):
                conflicts = line.split(":", 1)[1].strip()
            elif line.startswith("RELIABLE_CHUNKS:"):
                idx_str = line.split(":", 1)[1].strip()
                try:
                    reliable_indices = [int(x.strip()) - 1 for x in idx_str.split(",") if x.strip().isdigit()]
                except Exception:
                    reliable_indices = []

        return consistency, conflicts, reliable_indices
