"""
Analytics Engine — auto-generates insight reports from document chunks.

Computes: word frequency, chunk size distribution, top entities,
reading time, vocabulary richness, and sentence length stats.
No external dependencies beyond stdlib + numpy (already in requirements).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from .semantic_chunker import SemanticChunk
from .logger import get_logger

log = get_logger("analytics_engine")

# Common English stopwords (inline — no NLTK needed)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "as", "not", "no", "so", "if", "also",
    "than", "then", "when", "where", "which", "who", "what", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "there", "here", "about", "between", "own",
    "same", "just", "because", "while", "although", "however", "therefore",
    "thus", "hence", "since", "until", "unless", "whether", "though",
}


@dataclass
class ChunkStats:
    min_tokens: int
    max_tokens: int
    avg_tokens: float
    p50_tokens: float
    p90_tokens: float
    distribution: List[int]   # raw token counts, for histogram


@dataclass
class AnalyticsReport:
    filename: str
    total_words: int
    unique_words: int
    vocab_richness: float          # unique / total
    reading_time_min: float        # @ 200wpm
    top_words: List[Tuple[str, int]]   # top-20 (word, count)
    top_entities: List[str]            # capitalized n-grams
    chunk_stats: ChunkStats
    avg_sentence_len: float
    total_chunks: int


class AnalyticsEngine:
    """Computes rich analytics from a list of SemanticChunks."""

    def compute(self, chunks: List[SemanticChunk], filename: str) -> AnalyticsReport:
        if not chunks:
            return self._empty_report(filename)

        full_text = " ".join(c.text for c in chunks)
        words_raw = re.findall(r"\b[a-zA-Z]+\b", full_text)
        words_lower = [w.lower() for w in words_raw]

        total_words = len(words_lower)
        unique_words = len(set(words_lower))
        vocab_richness = round(unique_words / total_words, 3) if total_words else 0
        reading_time = round(total_words / 200, 1)

        # Top words (excluding stopwords, min length 3)
        filtered = [w for w in words_lower if w not in STOPWORDS and len(w) >= 3]
        top_words = Counter(filtered).most_common(20)

        # Top entities: capitalized phrases (2-3 words), not all-caps
        entity_pattern = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b")
        entities_raw = entity_pattern.findall(full_text)
        entity_counts = Counter(entities_raw)
        top_entities = [e for e, _ in entity_counts.most_common(15)]

        # Chunk size distribution
        token_counts = [c.token_count for c in chunks]
        sorted_tc = sorted(token_counts)
        n = len(sorted_tc)
        chunk_stats = ChunkStats(
            min_tokens=sorted_tc[0],
            max_tokens=sorted_tc[-1],
            avg_tokens=round(sum(sorted_tc) / n, 1),
            p50_tokens=float(sorted_tc[n // 2]),
            p90_tokens=float(sorted_tc[int(n * 0.9)]),
            distribution=sorted_tc,
        )

        # Avg sentence length
        sentences = re.split(r"[.!?]+", full_text)
        sentence_lengths = [
            len(re.findall(r"\b\w+\b", s)) for s in sentences if s.strip()
        ]
        avg_sentence_len = round(
            sum(sentence_lengths) / len(sentence_lengths), 1
        ) if sentence_lengths else 0

        log.info(f"Analytics computed for {filename}: {total_words}w, {len(chunks)} chunks")

        return AnalyticsReport(
            filename=filename,
            total_words=total_words,
            unique_words=unique_words,
            vocab_richness=vocab_richness,
            reading_time_min=reading_time,
            top_words=top_words,
            top_entities=top_entities,
            chunk_stats=chunk_stats,
            avg_sentence_len=avg_sentence_len,
            total_chunks=len(chunks),
        )

    def _empty_report(self, filename: str) -> AnalyticsReport:
        return AnalyticsReport(
            filename=filename, total_words=0, unique_words=0,
            vocab_richness=0, reading_time_min=0,
            top_words=[], top_entities=[],
            chunk_stats=ChunkStats(0, 0, 0, 0, 0, []),
            avg_sentence_len=0, total_chunks=0,
        )
