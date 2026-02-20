"""
Citation Highlighter & Document Comparator

New RAG features #3 and #4:

  Citation Highlighter:
    - Parses LLM answers for factual claims
    - Maps each claim back to the source chunk that contains it
    - Returns structured citations with inline markers [1], [2], …

  Document Comparator:
    - Given 2+ document IDs and a comparison question
    - Retrieves relevant chunks from EACH document independently
    - Asks the LLM to compare/contrast them side-by-side
    - Returns a structured comparison with per-document evidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from src.llm_provider import get_llm as _get_llm_provider
from langchain_core.messages import HumanMessage


# ── Citation structures ───────────────────────────────────────────────────────

@dataclass
class Citation:
    marker: str             # e.g. "[1]"
    claim: str              # the sentence this citation supports
    source_chunk_id: str
    source_filename: str
    source_version: int
    evidence: str           # the chunk text that supports the claim


@dataclass
class CitedAnswer:
    answer_with_markers: str    # answer text with [1], [2] inline
    plain_answer: str           # original answer without markers
    citations: List[Citation]
    uncited_sentences: List[str]  # sentences we couldn't find evidence for


# ── Citation highlighter ──────────────────────────────────────────────────────

CITATION_PROMPT = """Given an AI-generated answer and a list of source passages, your job is to:
1. Split the answer into individual factual claims (sentences).
2. For each claim, find the most supporting source passage (if any).
3. Return the answer with inline citation markers [1], [2], … added after each supported claim.
4. At the end, list citations in order: [N] filename | chunk_id | key evidence phrase.

Answer to cite:
{answer}

Source passages:
{sources}

Return the cited answer followed by a "CITATIONS:" section.
Format each citation as: [N]|||filename|||chunk_id|||evidence_phrase

Example:
The policy covers all employees [1]. Remote workers are included [2].
CITATIONS:
[1]|||policy.pdf|||doc_chunk_0002|||covers all employees
[2]|||policy.pdf|||doc_chunk_0005|||Remote workers are included
"""


class CitationHighlighter:
    """Adds inline citations to RAG answers by grounding claims in source chunks."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self._llm = _get_llm_provider(model=model, temperature=0)

    def highlight(self, answer: str, source_docs: List[Document]) -> CitedAnswer:
        if not source_docs or not answer.strip():
            return CitedAnswer(
                answer_with_markers=answer,
                plain_answer=answer,
                citations=[],
                uncited_sentences=[],
            )

        sources_text = "\n\n".join(
            f"[{d.metadata.get('chunk_id','?')}] {d.metadata.get('filename','?')} v{d.metadata.get('doc_version',1)}:\n{d.page_content[:400]}"
            for d in source_docs
        )

        prompt = CITATION_PROMPT.format(answer=answer, sources=sources_text)
        response = self._llm([HumanMessage(content=prompt)])
        raw = response.content.strip()

        return self._parse_response(raw, answer, source_docs)

    def _parse_response(self, raw: str, original_answer: str, docs: List[Document]) -> CitedAnswer:
        if "CITATIONS:" not in raw:
            return CitedAnswer(answer_with_markers=original_answer, plain_answer=original_answer, citations=[], uncited_sentences=[])

        parts = raw.split("CITATIONS:", 1)
        cited_answer = parts[0].strip()
        citations_block = parts[1].strip() if len(parts) > 1 else ""

        citations: List[Citation] = []
        marker_pattern = re.compile(r"\[(\d+)\]\|\|\|(.+?)\|\|\|(.+?)\|\|\|(.+)")
        for line in citations_block.split("\n"):
            m = marker_pattern.match(line.strip())
            if m:
                num, filename, chunk_id, evidence = m.groups()
                # Find matching doc
                matching_doc = next(
                    (d for d in docs if d.metadata.get("chunk_id") == chunk_id), None
                )
                citations.append(Citation(
                    marker=f"[{num}]",
                    claim="",
                    source_chunk_id=chunk_id,
                    source_filename=filename,
                    source_version=matching_doc.metadata.get("doc_version", 1) if matching_doc else 1,
                    evidence=evidence,
                ))

        return CitedAnswer(
            answer_with_markers=cited_answer,
            plain_answer=original_answer,
            citations=citations,
            uncited_sentences=[],
        )


# ── Document comparator ───────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    question: str
    doc_names: List[str]
    per_doc_evidence: Dict[str, List[str]]   # filename → relevant chunks
    comparison_table: str                     # markdown table
    narrative: str                            # prose comparison


COMPARE_PROMPT = """You are comparing information across multiple documents to answer the user's question.

Question: {question}

{doc_sections}

Instructions:
1. Write a markdown comparison table with rows = key aspects, columns = documents.
2. Write a 2-4 sentence narrative comparison highlighting agreements, differences, and which document has more detail.

Format your response as:
TABLE:
<markdown table here>

NARRATIVE:
<prose comparison here>
"""


class DocumentComparator:
    """Compares information across 2+ documents for a given question."""

    def __init__(self, knowledge_base, model: str = "gpt-3.5-turbo"):
        self.kb = knowledge_base
        self._llm = _get_llm_provider(model=model, temperature=0)

    def compare(
        self,
        question: str,
        doc_ids: List[str],
        doc_names: Dict[str, str],   # doc_id → filename
        k_per_doc: int = 3,
    ) -> ComparisonResult:
        per_doc_evidence: Dict[str, List[str]] = {}
        doc_sections = []

        for doc_id in doc_ids:
            fname = doc_names.get(doc_id, doc_id)
            results = self.kb.search(question, k=k_per_doc, doc_ids=[doc_id])
            chunks = [doc.page_content for doc, _ in results] if results and isinstance(results[0], tuple) else [d.page_content for d in results]
            per_doc_evidence[fname] = chunks
            joined = "\n---\n".join(chunks[:k_per_doc]) if chunks else "No relevant content found."
            doc_sections.append(f"=== {fname} ===\n{joined}")

        prompt = COMPARE_PROMPT.format(
            question=question,
            doc_sections="\n\n".join(doc_sections),
        )
        response = self._llm([HumanMessage(content=prompt)])
        raw = response.content.strip()

        table = narrative = ""
        if "TABLE:" in raw and "NARRATIVE:" in raw:
            t_part = raw.split("TABLE:", 1)[1]
            parts = t_part.split("NARRATIVE:", 1)
            table = parts[0].strip()
            narrative = parts[1].strip() if len(parts) > 1 else ""
        else:
            narrative = raw

        return ComparisonResult(
            question=question,
            doc_names=[doc_names.get(d, d) for d in doc_ids],
            per_doc_evidence=per_doc_evidence,
            comparison_table=table,
            narrative=narrative,
        )