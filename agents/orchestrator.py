"""
Agent 7 — Orchestrator Agent
Coordinates the entire multi-agent RAG pipeline with retry logic.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.config import config

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Final result returned by the orchestrator."""
    query: str
    answer: str
    sources: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    retrieved_chunks: int = 0
    filtered_chunks: int = 0
    consistency: str = "UNKNOWN"
    conflicts: str = ""
    key_facts: str = ""
    hallucination_verdict: str = "UNKNOWN"
    hallucination_confidence: float = 0.0
    is_hallucinated: bool = False
    retry_count: int = 0
    elapsed_seconds: float = 0.0
    has_context: bool = True
    agent_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "sub_queries": self.sub_queries,
            "retrieved_chunks": self.retrieved_chunks,
            "filtered_chunks": self.filtered_chunks,
            "consistency": self.consistency,
            "conflicts": self.conflicts,
            "key_facts": self.key_facts,
            "hallucination_verdict": self.hallucination_verdict,
            "hallucination_confidence": self.hallucination_confidence,
            "is_hallucinated": self.is_hallucinated,
            "retry_count": self.retry_count,
            "elapsed_seconds": self.elapsed_seconds,
            "has_context": self.has_context,
            "agent_trace": self.agent_trace,
        }

    @property
    def confidence_label(self) -> str:
        c = self.hallucination_confidence
        if c >= 0.80:
            return "🟢 HIGH"
        elif c >= 0.55:
            return "🟡 MEDIUM"
        else:
            return "🔴 LOW"


class OrchestratorAgent:
    """
    Coordinates all agents in the multi-agent RAG pipeline:

    Query → Reformulate → Retrieve → Filter → Cross-Reference
          → Generate Answer → Check Hallucination → Return Result
    """

    def __init__(self) -> None:
        # Lazy imports to avoid circular dependencies
        from agents.query_reformulation import QueryReformulationAgent
        from agents.retriever import RetrieverAgent
        from agents.relevance_filter import RelevanceFilterAgent
        from agents.cross_reference import CrossReferenceAgent
        from agents.answer_generator import AnswerGeneratorAgent
        from agents.hallucination_checker import HallucinationCheckerAgent

        logger.info("Initializing OrchestratorAgent and all sub-agents...")
        self._reformulator = QueryReformulationAgent()
        self._retriever = RetrieverAgent()
        self._filter = RelevanceFilterAgent()
        self._cross_ref = CrossReferenceAgent()
        self._generator = AnswerGeneratorAgent()
        self._hallucination = HallucinationCheckerAgent()
        logger.info("All agents initialized ✓")

    def run(self, query: str, max_retries: Optional[int] = None) -> RAGResult:
        """
        Execute the full multi-agent RAG pipeline.

        Args:
            query: User's question
            max_retries: Override config MAX_RETRIES

        Returns:
            RAGResult with answer, sources, and quality metrics
        """
        max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
        start = time.time()
        result = RAGResult(query=query, answer="")
        retry = 0

        while retry <= max_retries:
            try:
                result = self._execute_pipeline(query, result, retry)
                if not result.is_hallucinated:
                    break  # Grounded answer found
                logger.warning(
                    "[Orchestrator] Hallucination detected (attempt %d/%d) — retrying with stricter settings",
                    retry + 1,
                    max_retries + 1,
                )
                retry += 1
            except Exception as exc:
                logger.error("[Orchestrator] Pipeline error on attempt %d: %s", retry + 1, exc)
                result.answer = f"Pipeline error: {exc}"
                result.agent_trace.append(f"ERROR: {exc}")
                break

        result.retry_count = retry
        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "[Orchestrator] Done in %.2fs | verdict=%s | retries=%d",
            result.elapsed_seconds,
            result.hallucination_verdict,
            retry,
        )
        return result

    def _execute_pipeline(
        self, query: str, result: RAGResult, attempt: int
    ) -> RAGResult:
        trace = result.agent_trace

        # ── Step 1: Query Reformulation ──────────────────────────
        trace.append("🔄 Step 1: Query Reformulation")
        sub_queries = self._reformulator.run(query)
        # Include original query always
        if query not in sub_queries:
            sub_queries.insert(0, query)
        result.sub_queries = sub_queries
        logger.info("[Orchestrator] Sub-queries: %s", sub_queries)

        # ── Step 2: Retrieval ─────────────────────────────────────
        trace.append(f"📚 Step 2: Retrieval ({len(sub_queries)} queries)")
        k = config.TOP_K_CHUNKS + (attempt * 3)  # Expand on retry
        chunks = self._retriever.run(sub_queries, k=k)
        result.retrieved_chunks = len(chunks)

        if not chunks:
            result.answer = (
                "No relevant documents found in the knowledge base. "
                "Please upload relevant documents first using `python ingest.py`."
            )
            result.has_context = False
            result.hallucination_verdict = "NO_CONTEXT"
            trace.append("⚠️ No documents retrieved")
            return result

        # ── Step 3: Relevance Filtering ───────────────────────────
        trace.append(f"🔍 Step 3: Relevance Filtering ({len(chunks)} chunks)")
        threshold = max(1.0, config.RELEVANCE_THRESHOLD - (attempt * 1.5))  # Lower on retry
        filtered = self._filter.run(query, chunks, threshold=threshold)
        result.filtered_chunks = len(filtered)
        trace.append(f"  → {len(filtered)} chunks passed filter (threshold={threshold:.1f})")

        # ── Step 4: Cross-Reference ───────────────────────────────
        trace.append(f"🔗 Step 4: Cross-Reference ({len(filtered)} chunks)")
        verified, cross_report = self._cross_ref.run(query, filtered)
        result.consistency = cross_report.get("consistency", "UNKNOWN")
        result.conflicts = cross_report.get("conflicts", "")
        result.key_facts = cross_report.get("key_facts", "")
        trace.append(f"  → Consistency: {result.consistency}")

        # ── Step 5: Answer Generation ─────────────────────────────
        trace.append(f"✍️ Step 5: Answer Generation ({len(verified)} verified chunks)")
        gen_result = self._generator.run(query, verified, key_facts=result.key_facts)
        result.answer = gen_result["answer"]
        result.sources = gen_result["sources"]
        result.has_context = gen_result["has_context"]

        # ── Step 6: Hallucination Check ───────────────────────────
        trace.append("🔎 Step 6: Hallucination Check")
        hal_result = self._hallucination.run(result.answer, verified)
        result.hallucination_verdict = hal_result["verdict"]
        result.hallucination_confidence = hal_result["confidence"]
        result.is_hallucinated = hal_result["is_hallucinated"]
        trace.append(
            f"  → Verdict: {result.hallucination_verdict} | Confidence: {result.hallucination_confidence:.2f}"
        )

        return result
