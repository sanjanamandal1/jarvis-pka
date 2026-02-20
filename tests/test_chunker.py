"""Tests for SemanticChunker (no OpenAI calls required)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.semantic_chunker import split_into_sentences, SemanticChunk


def test_sentence_splitter_basic():
    text = "Hello world. This is a test. Does it work?"
    sents = split_into_sentences(text)
    assert len(sents) >= 1


def test_sentence_splitter_filters_short():
    text = "Hi. This is a full and complete sentence that is long enough."
    sents = split_into_sentences(text)
    # "Hi." should be filtered (< 15 chars)
    assert all(len(s) > 15 for s in sents)


def test_semantic_chunk_dataclass():
    chunk = SemanticChunk(
        chunk_id="doc_chunk_0000",
        text="This is a test chunk with some content.",
        sentences=["This is a test chunk with some content."],
        token_count=8,
    )
    d = chunk.to_dict()
    assert d["chunk_id"] == "doc_chunk_0000"
    assert d["token_count"] == 8
    assert "text" in d


def test_chunk_to_dict_no_embedding():
    chunk = SemanticChunk(
        chunk_id="c1",
        text="hello world",
        sentences=["hello world"],
    )
    d = chunk.to_dict()
    assert "embedding" not in d   # embedding excluded from dict
