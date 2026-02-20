"""Tests for TemporalVersionManager."""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.temporal_manager import TemporalVersionManager
from src.semantic_chunker import SemanticChunk


def make_chunks(n=3, doc_id="test"):
    return [
        SemanticChunk(
            chunk_id=f"{doc_id}_chunk_{i:04d}",
            text=f"This is chunk number {i} with some content about topic {i}.",
            sentences=[f"This is chunk number {i} with some content about topic {i}."],
            token_count=12,
        )
        for i in range(n)
    ]


@pytest.fixture
def tmp_manager(tmp_path):
    """TemporalVersionManager backed by a temp directory."""
    return TemporalVersionManager(persist_dir=str(tmp_path / "pka"))


def test_register_new_document(tmp_manager):
    chunks = make_chunks()
    doc_id, is_new, diff = tmp_manager.register_document("report.pdf", "Hello world content.", chunks)
    assert is_new is True
    assert diff is None
    assert doc_id is not None


def test_same_content_no_update(tmp_manager):
    chunks = make_chunks()
    text = "Same content every time."
    tmp_manager.register_document("doc.pdf", text, chunks)
    _, is_new, diff = tmp_manager.register_document("doc.pdf", text, chunks)
    assert is_new is False
    assert diff is None


def test_updated_content_creates_new_version(tmp_manager):
    chunks = make_chunks()
    tmp_manager.register_document("doc.pdf", "Version one content here.", chunks)
    _, is_new, diff = tmp_manager.register_document("doc.pdf", "Version two content here — different.", chunks)
    assert is_new is False
    assert diff is not None

    history = tmp_manager.get_document_history(
        tmp_manager._canonical_id("doc.pdf")
    )
    assert len(history) == 2


def test_get_temporal_context(tmp_manager):
    chunks = make_chunks()
    doc_id, _, _ = tmp_manager.register_document("notes.pdf", "Some notes content.", chunks)
    ctx = tmp_manager.get_temporal_context([doc_id])
    assert "notes.pdf" in ctx
