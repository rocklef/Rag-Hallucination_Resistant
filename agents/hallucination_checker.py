"""
Agent 6 — Hallucination Checker Agent
Uses NLI (Natural Language Inference) to verify that every claim in the
generated answer is entailed by the retrieved context.

Two modes:
  1. NLI model (cross-encoder/nli-deberta-v3-small) — local, high accuracy
  2. LLM-based fallback — used when transformers model not available
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

from core.config import config

logger = logging.getLogger(__name__)


# ── NLI-based checker ────────────────────────────────────────────────────────

class NLIHallucinationChecker:
    """Uses a cross-encoder NLI model to check entailment of answer vs context."""

    def __init__(self) -> None:
        from transformers import pipeline

        logger.info("Loading NLI model: %s", config.NLI_MODEL)
        self._nli = pipeline(
            "text-classification",
            model=config.NLI_MODEL,
            top_k=None,
        )

    def check(self, answer: str, context_chunks: List[str]) -> Tuple[float, str, List[Dict]]:
        """
        Returns (confidence_score, verdict, sentence_details).
        confidence_score: 0.0 (fully hallucinated) → 1.0 (fully grounded)
        verdict: "GROUNDED" | "PARTIALLY_GROUNDED" | "HALLUCINATED"
        """
        sentences = self._split_sentences(answer)
        if not sentences or not context_chunks:
            return 0.5, "UNCERTAIN", []

        context_blob = " ".join(context_chunks)[:4000]
        sentence_scores: List[Dict] = []

        for sent in sentences:
            if len(sent.strip()) < 10:
                continue
            input_pair = f"{context_blob} [SEP] {sent}"
            try:
                preds = self._nli(input_pair)[0]
                label_score = {p["label"].lower(): p["score"] for p in preds}
                entail = label_score.get("entailment", 0.0)
                contradict = label_score.get("contradiction", 0.0)
                neutral = label_score.get("neutral", 0.0)
                sentence_scores.append({
                    "sentence": sent,
                    "entailment": entail,
                    "contradiction": contradict,
                    "neutral": neutral,
                    "verdict": "GROUNDED" if entail > 0.5 else "UNCERTAIN" if neutral > contradict else "HALLUCINATED",
                })
            except Exception as exc:
                logger.warning("NLI check failed for sentence: %s", exc)
                sentence_scores.append({
                    "sentence": sent,
                    "entailment": 0.5,
                    "contradiction": 0.0,
                    "neutral": 0.5,
                    "verdict": "UNCERTAIN",
                })

        if not sentence_scores:
            return 0.5, "UNCERTAIN", []

        avg_entail = sum(s["entailment"] for s in sentence_scores) / len(sentence_scores)
        verdict = self._score_to_verdict(avg_entail)
        return avg_entail, verdict, sentence_scores

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        if score >= 0.75:
            return "GROUNDED"
        elif score >= 0.45:
            return "PARTIALLY_GROUNDED"
        else:
            return "HALLUCINATED"


# ── LLM-based fallback checker ───────────────────────────────────────────────

class LLMHallucinationChecker:
    """LLM-based hallucination checker (fallback when NLI model unavailable)."""

    def __init__(self) -> None:
        from langchain.prompts import PromptTemplate
        from agents.llm_factory import get_llm

        self._llm = get_llm(temperature=0.0)
        self._prompt = PromptTemplate(
            input_variables=["answer", "context"],
            template="""You are a hallucination detection expert. Evaluate whether the given answer 
is fully supported by the provided context.

Context:
\"\"\"
{context}
\"\"\"

Answer to evaluate:
\"\"\"
{answer}
\"\"\"

For each claim in the answer:
- Is it directly stated in the context? (GROUNDED)
- Is it implied but not directly stated? (UNCERTAIN)  
- Is it not present in the context? (HALLUCINATED)

Provide your evaluation in this exact format:
OVERALL_VERDICT: <GROUNDED|PARTIALLY_GROUNDED|HALLUCINATED>
CONFIDENCE: <0.0 to 1.0>
DETAILS: <brief explanation>""",
        )
        self._chain = self._prompt | self._llm

    def check(self, answer: str, context_chunks: List[str]) -> Tuple[float, str, List[Dict]]:
        context = " ".join(context_chunks)[:3000]
        try:
            result = self._chain.invoke({"answer": answer, "context": context})
            raw = result if isinstance(result, str) else result.content
            return self._parse(raw)
        except Exception as exc:
            logger.warning("[HallucinationChecker] LLM check failed: %s", exc)
            return 0.5, "UNCERTAIN", [{"sentence": answer, "verdict": "UNCERTAIN"}]

    @staticmethod
    def _parse(text: str) -> Tuple[float, str, List[Dict]]:
        verdict = "PARTIALLY_GROUNDED"
        confidence = 0.5
        details = ""
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("OVERALL_VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                if v in {"GROUNDED", "PARTIALLY_GROUNDED", "HALLUCINATED"}:
                    verdict = v
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("DETAILS:"):
                details = line.split(":", 1)[1].strip()
        return confidence, verdict, [{"sentence": "Full answer", "verdict": verdict, "details": details}]


# ── Public interface ──────────────────────────────────────────────────────────

class HallucinationCheckerAgent:
    """
    Main hallucination checker — tries NLI model first, falls back to LLM.
    """

    def __init__(self) -> None:
        self._checker = self._load_checker()

    def _load_checker(self):
        try:
            checker = NLIHallucinationChecker()
            logger.info("[HallucinationChecker] Using NLI model")
            return checker
        except Exception as exc:
            logger.warning("[HallucinationChecker] NLI model unavailable (%s) — using LLM fallback", exc)
            return LLMHallucinationChecker()

    def run(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check if the answer is hallucinated.

        Returns:
            {
                "confidence": float (0-1),
                "verdict": "GROUNDED"|"PARTIALLY_GROUNDED"|"HALLUCINATED",
                "is_hallucinated": bool,
                "sentence_details": List[dict],
            }
        """
        if not answer or not chunks:
            return {
                "confidence": 0.0,
                "verdict": "UNCERTAIN",
                "is_hallucinated": True,
                "sentence_details": [],
            }

        context_texts = [c["text"] for c in chunks]
        confidence, verdict, details = self._checker.check(answer, context_texts)

        is_hallucinated = verdict == "HALLUCINATED" or confidence < config.HALLUCINATION_THRESHOLD

        logger.info(
            "[HallucinationChecker] verdict=%s | confidence=%.3f | hallucinated=%s",
            verdict,
            confidence,
            is_hallucinated,
        )

        return {
            "confidence": confidence,
            "verdict": verdict,
            "is_hallucinated": is_hallucinated,
            "sentence_details": details,
        }
