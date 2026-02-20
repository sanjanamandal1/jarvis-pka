"""
Hybrid Search Engine — BM25 sparse retrieval + FAISS dense retrieval.

Combines keyword precision with semantic understanding using
Reciprocal Rank Fusion (RRF) to merge ranked result lists.

New RAG feature #1 of 4.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from langchain_core.documents import Document


# ── BM25 implementation (no external dependency) ─────────────────────────────

class BM25Index:
    """
    Lightweight BM25 index built from a list of LangChain Documents.
    
    BM25 scores documents based on term frequency (TF) normalized
    by document length, with inverse document frequency (IDF) weighting.
    Parameters k1=1.5, b=0.75 are standard Okapi BM25 defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Document] = []
        self.tf: List[Dict[str, float]] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.avg_dl: float = 0.0
        self.N: int = 0

    def fit(self, docs: List[Document]):
        """Index a list of Documents."""
        self.docs = docs
        self.N = len(docs)
        total_len = 0

        tokenized = []
        for doc in docs:
            tokens = self._tokenize(doc.page_content)
            tokenized.append(tokens)
            total_len += len(tokens)
            for tok in set(tokens):
                self.df[tok] += 1

        self.avg_dl = total_len / max(self.N, 1)

        for tokens in tokenized:
            freq: Dict[str, float] = defaultdict(float)
            for tok in tokens:
                freq[tok] += 1.0
            self.tf.append(dict(freq))

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Return top-k documents with BM25 scores."""
        if not self.docs:
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = []

        for i, (doc, tf_map) in enumerate(zip(self.docs, self.tf)):
            dl = sum(tf_map.values())
            score = 0.0
            for tok in query_tokens:
                if tok not in tf_map:
                    continue
                tf = tf_map[tok]
                idf = math.log((self.N - self.df[tok] + 0.5) / (self.df[tok] + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += idf * numerator / denominator
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.docs[i], s) for i, s in ranked[:k] if s > 0]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9]{1,}\b", text)
        stopwords = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "is","are","was","were","be","been","being","have","has","had","do",
            "does","did","will","would","could","should","may","might","this","that",
            "these","those","it","its","from","by","about","as","into","through",
        }
        return [t for t in tokens if t not in stopwords]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.
    chunk_id → fused score.
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 keyword search + FAISS semantic search via RRF.

    Usage
    -----
    retriever = HybridRetriever(vectorstore, alpha=0.5)
    retriever.fit(all_docs)
    results = retriever.search("what is the refund policy?", k=6)
    """

    def __init__(self, vectorstore, alpha: float = 0.5, bm25_k1: float = 1.5, bm25_b: float = 0.75):
        """
        alpha : weight for semantic score (0=BM25 only, 1=semantic only, 0.5=equal)
        """
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self._all_docs: List[Document] = []

    def fit(self, docs: List[Document]):
        """Build BM25 index from all documents."""
        self._all_docs = docs
        self.bm25.fit(docs)

    def search(self, query: str, k: int = 6, doc_ids: Optional[List[str]] = None) -> List[Document]:
        """Hybrid search returning top-k Documents."""
        fetch_k = k * 4

        # Dense retrieval
        dense_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)
        if doc_ids:
            dense_results = [(d, s) for d, s in dense_results if d.metadata.get("doc_id") in doc_ids]

        # Sparse retrieval
        sparse_pool = self._all_docs
        if doc_ids:
            sparse_pool = [d for d in self._all_docs if d.metadata.get("doc_id") in doc_ids]
        self.bm25.fit(sparse_pool)
        sparse_results = self.bm25.search(query, k=fetch_k)

        # Build id→doc map
        id_to_doc: Dict[str, Document] = {}
        for doc, _ in dense_results:
            cid = doc.metadata.get("chunk_id", doc.page_content[:40])
            id_to_doc[cid] = doc
        for doc, _ in sparse_results:
            cid = doc.metadata.get("chunk_id", doc.page_content[:40])
            id_to_doc[cid] = doc

        # RRF fusion
        dense_ranked = [(d.metadata.get("chunk_id", d.page_content[:40]), s) for d, s in dense_results]
        sparse_ranked = [(d.metadata.get("chunk_id", d.page_content[:40]), s) for d, s in sparse_results]
        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])

        results = []
        for chunk_id, _ in fused[:k]:
            if chunk_id in id_to_doc:
                results.append(id_to_doc[chunk_id])

        return results

    def as_langchain_retriever(self, k: int = 6, doc_ids: Optional[List[str]] = None):
        """Wrap as a LangChain BaseRetriever."""
        from langchain_core.retrievers import BaseRetriever
        from pydantic import Field

        outer = self

        class _HybridRetriever(BaseRetriever):
            _k: int = Field(default=k)
            _doc_ids: Optional[List[str]] = Field(default=doc_ids)

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return outer.search(query, k=self._k, doc_ids=self._doc_ids)

            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)

        return _HybridRetriever()