"""
Simple, reliable RAG chain.
"""
from __future__ import annotations
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from src.llm_provider import get_llm

PROMPT = """You are J.A.R.V.I.S., a helpful document assistant. Use the context below to answer the question.

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

Answer the question based on the context. Be detailed and helpful."""


class SimpleRAGChain:
    def __init__(self, kb, model=None, temporal_context="", **kwargs):
        self.kb = kb
        self.llm = get_llm(model=model, temperature=0)
        self.history = []

    def __call__(self, inputs: dict) -> dict:
        question = inputs["question"]

        # Search KB directly - most reliable approach
        docs = []
        try:
            results = self.kb.search(question, k=6)
            for item in results:
                if isinstance(item, tuple):
                    docs.append(item[0])
                elif hasattr(item, "page_content"):
                    docs.append(item)
        except Exception as e:
            pass

        if docs:
            context = "\n\n---\n\n".join(
                f"[{d.metadata.get('filename','?')}]\n{d.page_content}"
                for d in docs
            )
        else:
            context = "No documents found."

        prompt = PROMPT.format(context=context, question=question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "source_documents": docs}


def build_rag_chain(retriever=None, kb=None, model=None, temporal_context="", **kwargs):
    return SimpleRAGChain(kb=kb, model=model, temporal_context=temporal_context)


def format_sources(source_docs):
    seen = set()
    sources = []
    for doc in source_docs:
        key = doc.metadata.get("chunk_id", doc.page_content[:40])
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "filename": doc.metadata.get("filename", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "version": doc.metadata.get("doc_version", 1),
            "uploaded_at": doc.metadata.get("uploaded_at", ""),
            "text_preview": doc.page_content[:280] + ("..." if len(doc.page_content) > 280 else ""),
            "token_count": doc.metadata.get("token_count", 0),
        })
    return sources