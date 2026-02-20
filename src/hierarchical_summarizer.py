"""
Hierarchical Summarizer

Produces a 3-level summary tree for any document:

  Level 0 — chunk summaries   (1-2 sentences each)
  Level 1 — section summaries (groups of ~5 chunks)
  Level 2 — document summary  (full overview)

The hierarchy is stored alongside the vector index so that a query can
first search the document summary, then section, then chunk — providing
fast "zoom-in" retrieval without scanning every chunk.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.llm_provider import get_llm as _get_llm_provider
from langchain_core.messages import HumanMessage, SystemMessage

from .semantic_chunker import SemanticChunk


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ChunkSummary:
    chunk_id: str
    summary: str
    original_text: str

@dataclass
class SectionSummary:
    section_id: str
    summary: str
    chunk_ids: List[str]
    chunk_summaries: List[ChunkSummary]

@dataclass
class DocumentSummary:
    doc_id: str
    filename: str
    document_summary: str
    section_summaries: List[SectionSummary]
    chunk_summaries: List[ChunkSummary]
    total_chunks: int
    total_words: int

    def get_context_for_query(self, detail_level: str = "section") -> str:
        """Return summary text appropriate for a given detail level."""
        if detail_level == "document":
            return self.document_summary
        elif detail_level == "section":
            sections = "\n\n".join(
                f"[Section {i+1}]\n{s.summary}"
                for i, s in enumerate(self.section_summaries)
            )
            return f"Document Overview:\n{self.document_summary}\n\n---\n{sections}"
        else:  # chunk
            chunks = "\n\n".join(
                f"[{c.chunk_id}]\n{c.summary}"
                for c in self.chunk_summaries
            )
            return f"Document Overview:\n{self.document_summary}\n\n---\n{chunks}"


# ── Summarizer ───────────────────────────────────────────────────────────────

CHUNK_SUMMARY_PROMPT = """Summarize the following passage in 1-2 crisp sentences, preserving key facts, entities, and dates. 
Be specific — avoid vague phrases like "this section discusses".

Passage:
{text}

Summary:"""

SECTION_SUMMARY_PROMPT = """You have the following chunk-level summaries from a single section of a document.
Write a coherent 2-4 sentence section summary that captures the main ideas and their relationships.

Chunk summaries:
{chunk_summaries}

Section summary:"""

DOCUMENT_SUMMARY_PROMPT = """You have section-level summaries for an entire document.
Write an executive-style document summary (3-5 sentences) that answers:
- What is this document about?
- What are the key topics or findings?
- Who would find this useful and why?

Section summaries:
{section_summaries}

Document summary:"""


class HierarchicalSummarizer:
    """
    Builds a 3-level summary hierarchy for a list of SemanticChunks.
    
    Parameters
    ----------
    model : LLM model name (must support chat)
    chunks_per_section : how many chunks to group into one section
    """

    def __init__(self, model: str = "gpt-3.5-turbo", chunks_per_section: int = 5):
        self.model = model
        self.chunks_per_section = chunks_per_section
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = _get_llm_provider(model=self.model, temperature=0)
        return self._llm

    def _call_llm(self, prompt: str) -> str:
        llm = self._get_llm()
        response = llm([HumanMessage(content=prompt)])
        return response.content.strip()

    # ── Public API ───────────────────────────────────────────────────────────

    def summarize(self, chunks: List[SemanticChunk], doc_id: str, filename: str) -> DocumentSummary:
        """Build the full 3-level hierarchy. Returns a DocumentSummary."""

        # Level 0 — chunk summaries
        chunk_summaries = self._summarize_chunks(chunks)

        # Level 1 — section summaries
        section_summaries = self._summarize_sections(chunk_summaries, chunks)

        # Level 2 — document summary
        section_texts = "\n".join(f"- {s.summary}" for s in section_summaries)
        doc_summary = self._call_llm(
            DOCUMENT_SUMMARY_PROMPT.format(section_summaries=section_texts)
        )

        total_words = sum(len(c.text.split()) for c in chunks)

        return DocumentSummary(
            doc_id=doc_id,
            filename=filename,
            document_summary=doc_summary,
            section_summaries=section_summaries,
            chunk_summaries=chunk_summaries,
            total_chunks=len(chunks),
            total_words=total_words,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _summarize_chunks(self, chunks: List[SemanticChunk]) -> List[ChunkSummary]:
        summaries = []
        for chunk in chunks:
            # truncate very long chunks before sending to LLM
            text = textwrap.shorten(chunk.text, width=1500, placeholder=" …")
            summary = self._call_llm(CHUNK_SUMMARY_PROMPT.format(text=text))
            summaries.append(ChunkSummary(
                chunk_id=chunk.chunk_id,
                summary=summary,
                original_text=chunk.text,
            ))
        return summaries

    def _summarize_sections(
        self,
        chunk_summaries: List[ChunkSummary],
        chunks: List[SemanticChunk],
    ) -> List[SectionSummary]:
        sections = []
        n = len(chunk_summaries)
        for i in range(0, n, self.chunks_per_section):
            batch = chunk_summaries[i : i + self.chunks_per_section]
            combined = "\n".join(f"- {cs.summary}" for cs in batch)
            section_summary = self._call_llm(
                SECTION_SUMMARY_PROMPT.format(chunk_summaries=combined)
            )
            sections.append(SectionSummary(
                section_id=f"section_{i // self.chunks_per_section:03d}",
                summary=section_summary,
                chunk_ids=[cs.chunk_id for cs in batch],
                chunk_summaries=batch,
            ))
        return sections