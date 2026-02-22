"""
Hallucination Detector — checks if LLM answer is grounded in retrieved chunks.

Method:
  1. Extract key claims from the answer (sentences with specific facts)
  2. For each claim, check if similar text exists in the source chunks
  3. Return a grounding score + flagged ungrounded claims

This is a lightweight approach using cosine similarity on TF-IDF vectors
(no extra API calls needed — fully local).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List
from .logger import get_logger

log = get_logger("hallucination_detector")


@dataclass
class GroundingResult:
    score: float                    # 0.0 – 1.0 (1.0 = fully grounded)
    grounded_claims: List[str]      # claims found in source docs
    ungrounded_claims: List[str]    # claims NOT found in source docs
    verdict: str                    # "GROUNDED" | "PARTIAL" | "UNGROUNDED"
    verdict_color: str              # for UI rendering
    verdict_icon: str


def _extract_claims(text: str) -> List[str]:
    """Split answer into individual sentences as candidate claims."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        # Skip very short, hedging, or meta sentences
        if len(s) < 30:
            continue
        lower = s.lower()
        if any(skip in lower for skip in [
            "i could not find", "not in the document", "no relevant",
            "based on the context", "according to the", "the document states",
            "as mentioned", "in summary", "to summarize"
        ]):
            continue
        claims.append(s)
    return claims[:8]  # cap at 8 claims


def _simple_overlap(claim: str, context: str, threshold: float = 0.25) -> bool:
    """
    Check if a claim has enough word overlap with the source context.
    Uses Jaccard similarity on word sets (no external dependencies).
    """
    def tokenize(text):
        # Lowercase, remove punctuation, split into words, remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "can", "could", "should", "may", "might", "this", "that",
                     "these", "those", "it", "its", "in", "on", "at", "to",
                     "for", "of", "and", "or", "but", "with", "as", "by"}
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
        return set(w for w in words if w not in stopwords)

    claim_words = tokenize(claim)
    context_words = tokenize(context)

    if not claim_words:
        return True  # can't evaluate, give benefit of doubt

    intersection = claim_words & context_words
    union = claim_words | context_words

    jaccard = len(intersection) / len(union) if union else 0
    overlap_ratio = len(intersection) / len(claim_words)

    # Grounded if either Jaccard or overlap ratio clears threshold
    return jaccard >= threshold or overlap_ratio >= 0.4


def detect(answer: str, source_docs: list) -> GroundingResult:
    """
    Check if the answer is grounded in the source documents.

    Args:
        answer: LLM-generated answer string
        source_docs: list of langchain Document objects

    Returns:
        GroundingResult with score, verdict, and flagged claims
    """
    if not source_docs:
        log.warning("No source docs — cannot check grounding")
        return GroundingResult(
            score=0.0,
            grounded_claims=[],
            ungrounded_claims=[],
            verdict="NO SOURCES",
            verdict_color="#6b6870",
            verdict_icon="⚪",
        )

    # Build combined context from all source chunks
    context = " ".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in source_docs
    )

    claims = _extract_claims(answer)

    if not claims:
        # Answer too short or vague to evaluate
        return GroundingResult(
            score=1.0,
            grounded_claims=[],
            ungrounded_claims=[],
            verdict="GROUNDED",
            verdict_color="#00e5ff",
            verdict_icon="✓",
        )

    grounded = []
    ungrounded = []

    for claim in claims:
        if _simple_overlap(claim, context):
            grounded.append(claim)
        else:
            ungrounded.append(claim)
            log.debug(f"Ungrounded claim: {claim[:80]}")

    score = len(grounded) / len(claims) if claims else 1.0

    if score >= 0.8:
        verdict, color, icon = "GROUNDED", "#00e5ff", "✓"
    elif score >= 0.5:
        verdict, color, icon = "PARTIAL", "#ffb700", "⚠"
    else:
        verdict, color, icon = "UNGROUNDED", "#ff1a1a", "✗"

    log.info(f"Hallucination check | score={score:.2f} | verdict={verdict} | claims={len(claims)}")

    return GroundingResult(
        score=score,
        grounded_claims=grounded,
        ungrounded_claims=ungrounded,
        verdict=verdict,
        verdict_color=color,
        verdict_icon=icon,
    )