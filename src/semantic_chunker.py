"""
Semantic Chunker — Context-aware chunk boundaries using sentence transformers.

Instead of splitting at fixed token counts, we:
1. Encode every sentence with a sentence-transformer
2. Compute cosine similarity between adjacent sentences
3. Find "breakpoints" where similarity drops (topic shifts)
4. Merge sentences between breakpoints into coherent semantic chunks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# Lazy imports — only loaded when chunker is first used
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def split_into_sentences(text: str) -> List[str]:
    """Sentence splitter that handles abbreviations, bullets, and newlines."""
    # Normalize
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split on sentence-ending punctuation followed by whitespace + capital
    pattern = r"(?<=[.!?])\s+(?=[A-Z\"\'])|(?<=\n)\n+"
    raw = re.split(pattern, text)

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 15:  # drop micro-fragments
            sentences.append(s)
    return sentences


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of text with rich metadata."""
    chunk_id: str
    text: str
    sentences: List[str]
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    start_sentence_idx: int = 0
    end_sentence_idx: int = 0
    similarity_score: float = 1.0        # similarity to the chunk that follows
    source_file: str = ""
    page_hint: int = 0
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "sentences": self.sentences,
            "start_sentence_idx": self.start_sentence_idx,
            "end_sentence_idx": self.end_sentence_idx,
            "similarity_score": self.similarity_score,
            "source_file": self.source_file,
            "page_hint": self.page_hint,
            "token_count": self.token_count,
        }


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks.

    Algorithm:
    ----------
    1. Sentence-tokenize the document.
    2. Embed all sentences with a sentence-transformer.
    3. Compute adjacent-sentence cosine similarity.
    4. Apply percentile-based breakpoint detection.
    5. Merge consecutive sentences between breakpoints.
    6. Enforce min/max token bounds with re-splitting if needed.
    """

    def __init__(
        self,
        breakpoint_percentile: float = 85.0,   # lower → more chunks
        min_chunk_tokens: int = 80,
        max_chunk_tokens: int = 512,
        window_size: int = 2,                   # sentences to smooth over
    ):
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.window_size = window_size

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk(self, text: str, source_file: str = "", doc_id: str = "") -> List[SemanticChunk]:
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [self._make_chunk(sentences, 0, 0, source_file, doc_id, 0)]

        model = _get_model()
        embeddings = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)

        similarities = self._compute_windowed_similarities(embeddings)
        breakpoints = self._detect_breakpoints(similarities)

        chunks = self._build_chunks(sentences, embeddings, breakpoints, similarities, source_file, doc_id)
        chunks = self._enforce_size_constraints(chunks, model, source_file, doc_id)

        # Embed the final chunks for downstream use
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)
        for chunk, emb in zip(chunks, chunk_embeddings):
            chunk.embedding = emb

        return chunks

    # ── Internals ────────────────────────────────────────────────────────────

    def _compute_windowed_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Smooth similarity by averaging over a window of neighbours."""
        n = len(embeddings)
        similarities = []
        for i in range(n - 1):
            start = max(0, i - self.window_size + 1)
            end = min(n, i + self.window_size + 1)
            left = embeddings[start : i + 1].mean(axis=0)
            right = embeddings[i + 1 : end].mean(axis=0)
            similarities.append(cosine_similarity(left, right))
        return similarities

    def _detect_breakpoints(self, similarities: List[float]) -> List[int]:
        """Return indices (into sentences) where a new chunk should start."""
        if not similarities:
            return []
        threshold = float(np.percentile(similarities, 100 - self.breakpoint_percentile))
        return [i + 1 for i, s in enumerate(similarities) if s < threshold]

    def _build_chunks(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        breakpoints: List[int],
        similarities: List[float],
        source_file: str,
        doc_id: str,
    ) -> List[SemanticChunk]:
        segments = []
        prev = 0
        for bp in breakpoints:
            segments.append((prev, bp))
            prev = bp
        segments.append((prev, len(sentences)))

        chunks = []
        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = sentences[start:end]
            # boundary similarity = similarity between last sentence of this chunk and first of next
            sim = similarities[end - 1] if end - 1 < len(similarities) else 1.0
            chunks.append(self._make_chunk(seg_sentences, start, end - 1, source_file, doc_id, seg_idx, sim))
        return chunks

    def _make_chunk(
        self,
        sentences: List[str],
        start: int,
        end: int,
        source_file: str,
        doc_id: str,
        idx: int,
        similarity: float = 1.0,
    ) -> SemanticChunk:
        text = " ".join(sentences)
        return SemanticChunk(
            chunk_id=f"{doc_id}_chunk_{idx:04d}",
            text=text,
            sentences=sentences,
            start_sentence_idx=start,
            end_sentence_idx=end,
            similarity_score=similarity,
            source_file=source_file,
            token_count=len(text.split()),
        )

    def _enforce_size_constraints(
        self,
        chunks: List[SemanticChunk],
        model,
        source_file: str,
        doc_id: str,
    ) -> List[SemanticChunk]:
        """Merge tiny chunks and re-split oversized ones."""
        # Merge tiny chunks
        merged: List[SemanticChunk] = []
        buffer_sentences: List[str] = []
        buffer_start = 0

        for chunk in chunks:
            if chunk.token_count < self.min_chunk_tokens and merged or buffer_sentences:
                buffer_sentences.extend(chunk.sentences)
            else:
                if buffer_sentences:
                    merged.append(self._make_chunk(
                        buffer_sentences, buffer_start, buffer_start + len(buffer_sentences) - 1,
                        source_file, doc_id, len(merged)
                    ))
                    buffer_sentences = []
                if chunk.token_count < self.min_chunk_tokens:
                    buffer_sentences = chunk.sentences[:]
                    buffer_start = chunk.start_sentence_idx
                else:
                    merged.append(chunk)

        if buffer_sentences:
            if merged:
                last = merged[-1]
                merged[-1] = self._make_chunk(
                    last.sentences + buffer_sentences,
                    last.start_sentence_idx, last.end_sentence_idx + len(buffer_sentences),
                    source_file, doc_id, len(merged) - 1
                )
            else:
                merged.append(self._make_chunk(buffer_sentences, 0, len(buffer_sentences) - 1,
                                               source_file, doc_id, 0))

        # Re-split oversized chunks by sentence halving
        final: List[SemanticChunk] = []
        for chunk in merged:
            if chunk.token_count > self.max_chunk_tokens and len(chunk.sentences) > 1:
                mid = len(chunk.sentences) // 2
                for part_idx, part in enumerate([chunk.sentences[:mid], chunk.sentences[mid:]]):
                    final.append(self._make_chunk(
                        part, chunk.start_sentence_idx, chunk.end_sentence_idx,
                        source_file, doc_id, len(final)
                    ))
            else:
                final.append(chunk)

        # Re-number IDs
        for i, c in enumerate(final):
            c.chunk_id = f"{doc_id}_chunk_{i:04d}"

        return final
