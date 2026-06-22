"""
Unit tests for the Multi-Agent RAG system.
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── Core tests ────────────────────────────────────────────────────────────────

class TestConfig:
    def test_config_loads(self):
        from core.config import config
        assert config.LLM_PROVIDER in {"openai", "ollama"}
        assert config.TOP_K_CHUNKS > 0
        assert 0 < config.RELEVANCE_THRESHOLD <= 10
        assert 0 < config.HALLUCINATION_THRESHOLD <= 1

    def test_embedding_model_name(self):
        from core.config import config
        assert config.EMBEDDING_MODEL


class TestDocumentLoader:
    def test_load_text(self):
        from core.document_loader import load_text
        docs = load_text("This is a test document. It contains multiple sentences for testing.")
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all("text" in d and "metadata" in d for d in docs)

    def test_load_text_empty(self):
        from core.document_loader import load_text
        docs = load_text("")
        assert docs == []

    def test_load_sample_docs(self):
        from core.document_loader import load_directory
        sample_dir = Path(__file__).parent.parent / "data" / "sample_docs"
        if sample_dir.exists():
            docs = load_directory(sample_dir)
            assert len(docs) > 0


# ── Agent tests (mocked LLM) ──────────────────────────────────────────────────

class TestQueryReformulationAgent:
    def test_parse_output(self):
        from agents.query_reformulation import QueryReformulationAgent
        raw = "1. What is retrieval augmented generation?\n2. How does RAG work?\n3. Benefits of using RAG"
        parsed = QueryReformulationAgent._parse(raw)
        assert len(parsed) == 3
        assert "What is retrieval augmented generation?" in parsed

    def test_parse_bullet_output(self):
        from agents.query_reformulation import QueryReformulationAgent
        raw = "- What is retrieval augmented generation?\n- How does RAG work?\n- RAG benefits"
        parsed = QueryReformulationAgent._parse(raw)
        assert len(parsed) == 3

    def test_parse_empty(self):
        from agents.query_reformulation import QueryReformulationAgent
        parsed = QueryReformulationAgent._parse("")
        assert parsed == []


class TestRelevanceFilterAgent:
    def test_parse_score_numeric(self):
        from agents.relevance_filter import RelevanceFilterAgent
        assert RelevanceFilterAgent._parse_score("8") == 8.0
        assert RelevanceFilterAgent._parse_score("7.5") == 7.5
        assert RelevanceFilterAgent._parse_score("10") == 10.0

    def test_parse_score_clamped(self):
        from agents.relevance_filter import RelevanceFilterAgent
        assert RelevanceFilterAgent._parse_score("15") == 10.0
        assert RelevanceFilterAgent._parse_score("-1") == 0.0

    def test_parse_score_fallback(self):
        from agents.relevance_filter import RelevanceFilterAgent
        assert RelevanceFilterAgent._parse_score("N/A") == 5.0


class TestCrossReferenceAgent:
    def test_parse_conflict_response(self):
        from agents.cross_reference import CrossReferenceAgent
        raw = "CONSISTENCY: HIGH\nCONFLICTS: None detected\nRELIABLE_CHUNKS: 1,2,3"
        consistency, conflicts, indices = CrossReferenceAgent._parse_conflict_response(raw)
        assert consistency == "HIGH"
        assert conflicts == "None detected"
        assert indices == [0, 1, 2]

    def test_parse_conflict_low(self):
        from agents.cross_reference import CrossReferenceAgent
        raw = "CONSISTENCY: LOW\nCONFLICTS: Sources disagree on dates\nRELIABLE_CHUNKS: 2"
        consistency, conflicts, indices = CrossReferenceAgent._parse_conflict_response(raw)
        assert consistency == "LOW"
        assert "dates" in conflicts
        assert indices == [1]

    def test_format_chunks(self):
        from agents.cross_reference import CrossReferenceAgent
        chunks = [
            {"text": "Hello world", "metadata": {"source": "test.txt"}},
            {"text": "Foo bar", "metadata": {"source": "other.txt"}},
        ]
        formatted = CrossReferenceAgent._format_chunks(chunks)
        assert "[Chunk 1]" in formatted
        assert "[Chunk 2]" in formatted


class TestHallucinationChecker:
    def test_llm_checker_parse_grounded(self):
        from agents.hallucination_checker import LLMHallucinationChecker
        raw = "OVERALL_VERDICT: GROUNDED\nCONFIDENCE: 0.92\nDETAILS: All facts supported."
        conf, verdict, details = LLMHallucinationChecker._parse(raw)
        assert verdict == "GROUNDED"
        assert conf == pytest.approx(0.92)

    def test_llm_checker_parse_hallucinated(self):
        from agents.hallucination_checker import LLMHallucinationChecker
        raw = "OVERALL_VERDICT: HALLUCINATED\nCONFIDENCE: 0.2\nDETAILS: Fabricated info."
        conf, verdict, details = LLMHallucinationChecker._parse(raw)
        assert verdict == "HALLUCINATED"
        assert conf == pytest.approx(0.2)

    def test_nli_checker_split_sentences(self):
        from agents.hallucination_checker import NLIHallucinationChecker
        text = "This is sentence one. This is sentence two. And this is three."
        sentences = NLIHallucinationChecker._split_sentences(text)
        assert len(sentences) == 3

    def test_nli_score_to_verdict(self):
        from agents.hallucination_checker import NLIHallucinationChecker
        assert NLIHallucinationChecker._score_to_verdict(0.9) == "GROUNDED"
        assert NLIHallucinationChecker._score_to_verdict(0.6) == "PARTIALLY_GROUNDED"
        assert NLIHallucinationChecker._score_to_verdict(0.2) == "HALLUCINATED"


class TestOrchestratorResult:
    def test_result_confidence_label(self):
        from agents.orchestrator import RAGResult
        r = RAGResult(query="test", answer="test")
        r.hallucination_confidence = 0.9
        assert "HIGH" in r.confidence_label
        r.hallucination_confidence = 0.6
        assert "MEDIUM" in r.confidence_label
        r.hallucination_confidence = 0.3
        assert "LOW" in r.confidence_label

    def test_result_to_dict(self):
        from agents.orchestrator import RAGResult
        r = RAGResult(query="q", answer="a", sources=["s1"])
        d = r.to_dict()
        assert d["query"] == "q"
        assert d["answer"] == "a"
        assert d["sources"] == ["s1"]
