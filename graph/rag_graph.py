"""
LangGraph state machine for the Multi-Agent RAG pipeline.

State flows through nodes:
  reformulate → retrieve → filter → cross_reference → generate → hallucination_check → END
                                                                       ↑_______________|
                                                               (retry if hallucinated)
"""
from __future__ import annotations

import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ── State Schema ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    # Input
    query: str
    max_retries: int
    attempt: int

    # Agent outputs
    sub_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    filtered_chunks: List[Dict[str, Any]]
    verified_chunks: List[Dict[str, Any]]
    cross_ref_report: Dict[str, Any]
    answer: str
    sources: List[str]
    has_context: bool

    # Quality metrics
    hallucination_verdict: str
    hallucination_confidence: float
    is_hallucinated: bool
    sentence_details: List[Dict[str, Any]]

    # Trace
    agent_trace: Annotated[List[str], operator.add]


# ── Node functions ────────────────────────────────────────────────────────────

def node_reformulate(state: RAGState) -> RAGState:
    """Node 1: Reformulate the query into sub-queries."""
    from agents.query_reformulation import QueryReformulationAgent
    agent = QueryReformulationAgent()
    sub_queries = agent.run(state["query"])
    if state["query"] not in sub_queries:
        sub_queries.insert(0, state["query"])
    return {
        **state,
        "sub_queries": sub_queries,
        "agent_trace": [f"🔄 Reformulated into {len(sub_queries)} queries"],
    }


def node_retrieve(state: RAGState) -> RAGState:
    """Node 2: Multi-query retrieval from vector store."""
    from agents.retriever import RetrieverAgent
    from core.config import config
    agent = RetrieverAgent()
    k = config.TOP_K_CHUNKS + (state.get("attempt", 0) * 3)
    chunks = agent.run(state["sub_queries"], k=k)
    return {
        **state,
        "retrieved_chunks": chunks,
        "agent_trace": [f"📚 Retrieved {len(chunks)} unique chunks"],
    }


def node_filter(state: RAGState) -> RAGState:
    """Node 3: Relevance filtering."""
    from agents.relevance_filter import RelevanceFilterAgent
    from core.config import config
    agent = RelevanceFilterAgent()
    attempt = state.get("attempt", 0)
    threshold = max(1.0, config.RELEVANCE_THRESHOLD - (attempt * 1.5))
    filtered = agent.run(state["query"], state["retrieved_chunks"], threshold=threshold)
    return {
        **state,
        "filtered_chunks": filtered,
        "agent_trace": [f"🔍 {len(filtered)} chunks passed filter (threshold={threshold:.1f})"],
    }


def node_cross_reference(state: RAGState) -> RAGState:
    """Node 4: Cross-reference and verify facts."""
    from agents.cross_reference import CrossReferenceAgent
    agent = CrossReferenceAgent()
    verified, report = agent.run(state["query"], state["filtered_chunks"])
    return {
        **state,
        "verified_chunks": verified,
        "cross_ref_report": report,
        "agent_trace": [f"🔗 Cross-reference: {report.get('consistency', 'UNKNOWN')}"],
    }


def node_generate(state: RAGState) -> RAGState:
    """Node 5: Generate grounded answer."""
    from agents.answer_generator import AnswerGeneratorAgent
    agent = AnswerGeneratorAgent()
    key_facts = state.get("cross_ref_report", {}).get("key_facts", "")
    result = agent.run(state["query"], state["verified_chunks"], key_facts=key_facts)
    return {
        **state,
        "answer": result["answer"],
        "sources": result["sources"],
        "has_context": result["has_context"],
        "agent_trace": ["✍️ Answer generated"],
    }


def node_hallucination_check(state: RAGState) -> RAGState:
    """Node 6: Hallucination check."""
    from agents.hallucination_checker import HallucinationCheckerAgent
    agent = HallucinationCheckerAgent()
    result = agent.run(state["answer"], state["verified_chunks"])
    return {
        **state,
        "hallucination_verdict": result["verdict"],
        "hallucination_confidence": result["confidence"],
        "is_hallucinated": result["is_hallucinated"],
        "sentence_details": result["sentence_details"],
        "agent_trace": [
            f"🔎 Hallucination: {result['verdict']} (confidence={result['confidence']:.2f})"
        ],
    }


def should_retry(state: RAGState) -> str:
    """Conditional edge: retry if hallucinated and attempts remain."""
    if state.get("is_hallucinated") and state.get("attempt", 0) < state.get("max_retries", 2):
        return "retry"
    return "end"


def node_retry_increment(state: RAGState) -> RAGState:
    """Increment attempt counter for retry loop."""
    return {
        **state,
        "attempt": state.get("attempt", 0) + 1,
        "agent_trace": [f"⚠️ Retrying (attempt {state.get('attempt', 0) + 2})"],
    }


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("reformulate", node_reformulate)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("filter", node_filter)
    graph.add_node("cross_reference", node_cross_reference)
    graph.add_node("generate", node_generate)
    graph.add_node("hallucination_check", node_hallucination_check)
    graph.add_node("retry_increment", node_retry_increment)

    # Define edges
    graph.set_entry_point("reformulate")
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "cross_reference")
    graph.add_edge("cross_reference", "generate")
    graph.add_edge("generate", "hallucination_check")

    # Conditional edge: retry or end
    graph.add_conditional_edges(
        "hallucination_check",
        should_retry,
        {
            "retry": "retry_increment",
            "end": END,
        },
    )
    graph.add_edge("retry_increment", "retrieve")  # Re-retrieve with wider net

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_rag_graph():
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Building LangGraph RAG state machine...")
        _compiled_graph = build_rag_graph()
    return _compiled_graph


def run_graph(query: str, max_retries: int = 2) -> RAGState:
    """Run the full RAG graph for a query and return the final state."""
    graph = get_rag_graph()
    initial_state: RAGState = {
        "query": query,
        "max_retries": max_retries,
        "attempt": 0,
        "sub_queries": [],
        "retrieved_chunks": [],
        "filtered_chunks": [],
        "verified_chunks": [],
        "cross_ref_report": {},
        "answer": "",
        "sources": [],
        "has_context": True,
        "hallucination_verdict": "UNKNOWN",
        "hallucination_confidence": 0.0,
        "is_hallucinated": False,
        "sentence_details": [],
        "agent_trace": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state
