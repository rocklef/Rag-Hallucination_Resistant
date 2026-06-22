"""
Agent 5 — Answer Generator Agent
Generates a grounded, cited answer using ONLY the verified context.
Explicitly instructed NOT to hallucinate.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate

from agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = PromptTemplate(
    input_variables=["query", "context", "key_facts"],
    template="""You are a precise, trustworthy AI assistant. Answer the user's question using 
ONLY the provided context. Do not add any information not present in the context.

CRITICAL RULES:
1. Only use facts from the provided context — do NOT fabricate information
2. If the context does not contain enough information, say so explicitly
3. Cite your sources using [Chunk N] notation when making specific claims
4. Be concise and precise — avoid speculation or inference beyond the text
5. If you are uncertain about something, say "Based on the available context..."

Key Verified Facts (cross-referenced):
{key_facts}

Retrieved Context:
{context}

User Question: {query}

Answer (cite sources, be factual, no hallucination):""",
)

_FALLBACK_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""The user asked: {query}

Unfortunately, I was unable to find relevant information in the knowledge base to answer this question.

Please let the user know:
1. What information was NOT found
2. What they could try instead (rephrase, upload more documents, etc.)

Response:""",
)


class AnswerGeneratorAgent:
    """Generates grounded, citation-rich answers from verified context."""

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.1)
        self._chain = _ANSWER_PROMPT | self._llm
        self._fallback_chain = _FALLBACK_PROMPT | self._llm

    def run(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        key_facts: str = "",
    ) -> Dict[str, Any]:
        """
        Generate an answer from verified context.

        Returns:
            {
                "answer": str,
                "sources": List[str],
                "has_context": bool,
            }
        """
        if not chunks:
            logger.warning("[AnswerGenerator] No context — generating fallback response")
            answer = self._generate_fallback(query)
            return {"answer": answer, "sources": [], "has_context": False}

        context = self._format_context(chunks)
        sources = list({c.get("metadata", {}).get("source", "unknown") for c in chunks})

        logger.info("[AnswerGenerator] Generating answer from %d chunks", len(chunks))
        try:
            result = self._chain.invoke({
                "query": query,
                "context": context,
                "key_facts": key_facts or "Not available.",
            })
            answer = result if isinstance(result, str) else result.content
            logger.info("[AnswerGenerator] Answer generated (%d chars)", len(answer))
            return {"answer": answer.strip(), "sources": sources, "has_context": True}
        except Exception as exc:
            logger.error("[AnswerGenerator] Generation failed: %s", exc)
            return {
                "answer": f"An error occurred during answer generation: {exc}",
                "sources": sources,
                "has_context": True,
            }

    def _generate_fallback(self, query: str) -> str:
        try:
            result = self._fallback_chain.invoke({"query": query})
            return result if isinstance(result, str) else result.content
        except Exception:
            return (
                "I couldn't find relevant information to answer your question. "
                "Please try rephrasing or upload relevant documents."
            )

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        """Format verified chunks as numbered context blocks."""
        parts = []
        for i, chunk in enumerate(chunks[:8], 1):  # Cap at 8 chunks
            source = chunk.get("metadata", {}).get("source", "unknown")
            score = chunk.get("relevance_score", chunk.get("score", 0))
            parts.append(
                f"[Chunk {i}] (Source: {source} | Relevance: {score:.1f})\n{chunk['text'][:800]}"
            )
        return "\n\n---\n\n".join(parts)
