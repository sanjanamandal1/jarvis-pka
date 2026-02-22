"""
Personal Knowledge Base — manages FAISS index with rich chunk metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .semantic_chunker import SemanticChunk
from src.logger import get_logger

log = get_logger("knowledge_base")


INDEX_DIR = Path(".pka_data/faiss_index")


def _get_embeddings():
    """Always get fresh embeddings from current provider — called lazily."""
    from src.llm_provider import get_embeddings
    return get_embeddings()


class KnowledgeBase:
    def __init__(self):
        self._vectorstore: Optional[FAISS] = None
        self._chunk_meta: Dict[str, dict] = {}
        # NOTE: do NOT init embeddings here — provider may not be set yet
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(
        self,
        chunks: List[SemanticChunk],
        doc_id: str,
        doc_version: int,
        filename: str,
        uploaded_at: str,
    ):
        """Add (or replace) a document in the knowledge base."""
        log.info(f"Adding document: {filename} | v{doc_version} | {len(chunks)} chunks")
        self._remove_doc(doc_id)

        embeddings = _get_embeddings()  # lazy — uses current provider
        docs = []
        for chunk in chunks:
            meta = {
                "doc_id": doc_id,
                "doc_version": doc_version,
                "filename": filename,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "token_count": chunk.token_count,
                "uploaded_at": uploaded_at,
                "start_sentence_idx": chunk.start_sentence_idx,
                "end_sentence_idx": chunk.end_sentence_idx,
                "similarity_score": chunk.similarity_score,
            }
            self._chunk_meta[chunk.chunk_id] = meta
            docs.append(Document(page_content=chunk.text, metadata=meta))

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(docs, embeddings)
        else:
            self._vectorstore.add_documents(docs)

        self._save()

    def remove_document(self, doc_id: str):
        self._remove_doc(doc_id)
        self._save()

    def search(
        self,
        query: str,
        k: int = 6,
        doc_ids: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        if self._vectorstore is None:
            return []

        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k * 3)
        filtered = []
        for doc, score in results:
            if score < min_score:
                continue
            if doc_ids and doc.metadata.get("doc_id") not in doc_ids:
                continue
            filtered.append((doc, score))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:k]

    def get_retriever(self, k: int = 6, doc_ids: Optional[List[str]] = None):
        if self._vectorstore is None:
            return None

        search_kwargs = {"k": k}
        if doc_ids:
            search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}

        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def get_stats(self) -> dict:
        n_chunks = len(self._chunk_meta)
        doc_ids = {m["doc_id"] for m in self._chunk_meta.values()}
        return {
            "total_chunks": n_chunks,
            "total_documents": len(doc_ids),
            "doc_ids": list(doc_ids),
        }

    def is_empty(self) -> bool:
        return self._vectorstore is None or len(self._chunk_meta) == 0

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        if self._vectorstore:
            self._vectorstore.save_local(str(INDEX_DIR))
        meta_path = INDEX_DIR / "chunk_meta.json"
        meta_path.write_text(json.dumps(self._chunk_meta, indent=2), encoding="utf-8")

    def load(self):
        """Call this AFTER provider is configured to load persisted index."""
        meta_path = INDEX_DIR / "chunk_meta.json"
        if meta_path.exists():
            try:
                self._chunk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._chunk_meta = {}

        index_file = INDEX_DIR / "index.faiss"
        if index_file.exists():
            try:
                embeddings = _get_embeddings()
                self._vectorstore = FAISS.load_local(
                    str(INDEX_DIR), embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                self._vectorstore = None

    def _remove_doc(self, doc_id: str):
        self._chunk_meta = {
            cid: meta for cid, meta in self._chunk_meta.items()
            if meta.get("doc_id") != doc_id
        }