"""
Context-aware RAG chain with query classification.
Detects intent (summary/comparison/definition/factual/etc.)
and uses a tailored prompt for each.
"""
from __future__ import annotations
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from src.llm_provider import get_llm
from src.query_classifier import classify, ClassifiedQuery
from src.logger import get_logger

log = get_logger("rag_chain")


def _format_docs(docs):
    if not docs:
        return "No documents retrieved."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('filename','?')}]\n{d.page_content}"
        for d in docs
    )


def _format_history(history: list) -> str:
    lines = []
    for msg in history[-6:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines) if lines else "None"


class SimpleRAGChain:
    def __init__(self, kb, model=None, temporal_context="", **kwargs):
        self.kb = kb
        self.llm = get_llm(model=model, temperature=0)
        self.history = []

    def __call__(self, inputs: dict) -> dict:
        question = inputs["question"]

        # Step 1: Classify query intent
        classified: ClassifiedQuery = classify(question)
        log.info(f"Intent: {classified.intent} | Q: {question[:60]}")

        # Step 2: Retrieve relevant docs
        docs = []
        try:
            results = self.kb.search(question, k=6)
            for item in results:
                if isinstance(item, tuple):
                    docs.append(item[0])
                elif hasattr(item, "page_content"):
                    docs.append(item)
        except Exception as e:
            log.error(f"Retrieval error: {e}")

        context = _format_docs(docs)

        # Step 3: Build prompt using intent-specific template
        prompt = classified.prompt_template.format(
            context=context,
            question=question,
        )

        # Step 4: Call LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)

        log.info(f"Answer generated | intent={classified.intent} | len={len(answer)}")

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "source_documents": docs,
            "intent": classified.intent,
            "intent_icon": classified.icon,
        }


def build_rag_chain(retriever=None, kb=None, model=None, temporal_context="", **kwargs):
    return SimpleRAGChain(kb=kb, model=model, temporal_context=temporal_context)


def format_sources(source_docs: List[Document]) -> List[Dict[str, Any]]:
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