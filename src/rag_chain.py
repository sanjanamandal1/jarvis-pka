"""
Context-aware RAG chain with temporal awareness.
"""

from __future__ import annotations

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from src.llm_provider import get_llm


SYSTEM_TEMPLATE = """You are J.A.R.V.I.S., a Personal Knowledge Assistant with deep expertise in the user's uploaded documents.

{temporal_context}

{hierarchy_context}

INSTRUCTIONS:
- Answer ONLY from the provided context below. If the answer is not in the context, say "I could not find that in your documents."
- Cite the source filename when referencing specific facts.
- If documents have multiple versions, prefer the latest.
- Be concise but complete.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""


def _format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('filename','?')} | Chunk: {d.metadata.get('chunk_id','?')}]\n{d.page_content}"
        for d in docs
    )


def _format_history(history: list) -> str:
    lines = []
    for msg in history[-6:]:  # last 3 exchanges
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines) if lines else "None"


class SimpleRAGChain:
    """
    Simple RAG chain that works with any LangChain-compatible LLM.
    Avoids ConversationalRetrievalChain which has version compatibility issues.
    """

    def __init__(self, retriever, model=None, temporal_context="", hierarchy_context="", memory_window=5):
        self.retriever = retriever
        self.temporal_context = temporal_context
        self.hierarchy_context = hierarchy_context
        self.memory_window = memory_window
        self.llm = get_llm(model=model, temperature=0)
        self.history = []

    def __call__(self, inputs: dict) -> dict:
        question = inputs["question"]
        chat_history_str = _format_history(self.history)

        # Retrieve relevant docs
        try:
            docs = self.retriever.get_relevant_documents(question)
        except Exception:
            docs = []

        context = _format_docs(docs) if docs else "No relevant documents found."

        prompt = SYSTEM_TEMPLATE.format(
            temporal_context=self.temporal_context or "N/A",
            hierarchy_context=self.hierarchy_context or "",
            context=context,
            chat_history=chat_history_str,
            question=question,
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "source_documents": docs,
        }


def build_rag_chain(
    retriever,
    model: str = None,
    temporal_context: str = "",
    hierarchy_context: str = "",
    memory_window: int = 5,
):
    return SimpleRAGChain(
        retriever=retriever,
        model=model,
        temporal_context=temporal_context,
        hierarchy_context=hierarchy_context,
        memory_window=memory_window,
    )


def format_sources(source_docs: List[Document]) -> List[Dict[str, Any]]:
    seen = set()
    sources = []
    for doc in source_docs:
        key = doc.metadata.get("chunk_id", "")
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