"""Tests for AnalyticsEngine."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analytics_engine import AnalyticsEngine
from src.semantic_chunker import SemanticChunk


def make_chunks(texts):
    return [
        SemanticChunk(
            chunk_id=f"doc_chunk_{i:04d}",
            text=t,
            sentences=[t],
            token_count=len(t.split()),
        )
        for i, t in enumerate(texts)
    ]


def test_basic_report():
    chunks = make_chunks([
        "Artificial intelligence is transforming many industries including healthcare and finance.",
        "Machine learning models require large amounts of training data to perform well.",
        "Deep learning has revolutionized computer vision and natural language processing tasks.",
    ])
    engine = AnalyticsEngine()
    report = engine.compute(chunks, "test.pdf")
    assert report.total_words > 0
    assert report.unique_words > 0
    assert report.total_chunks == 3
    assert len(report.top_words) > 0
    assert report.reading_time_min >= 0


def test_vocab_richness_in_range():
    chunks = make_chunks([
        "The the the the the cat sat on the mat the.",
    ])
    engine = AnalyticsEngine()
    report = engine.compute(chunks, "repetitive.txt")
    assert 0.0 <= report.vocab_richness <= 1.0


def test_empty_chunks():
    engine = AnalyticsEngine()
    report = engine.compute([], "empty.pdf")
    assert report.total_words == 0
    assert report.total_chunks == 0


def test_chunk_stats_populated():
    chunks = make_chunks([
        "Short chunk here.",
        "This is a much longer chunk with many more words in it to test the statistics properly and ensure accuracy.",
    ])
    engine = AnalyticsEngine()
    report = engine.compute(chunks, "mixed.pdf")
    assert report.chunk_stats.min_tokens <= report.chunk_stats.max_tokens
    assert report.chunk_stats.avg_tokens > 0
