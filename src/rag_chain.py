"""
Context-aware RAG chain with temporal awareness.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from src.llm_provider import get_llm


SYSTEM_TEMPLATE = """You are a Personal Knowledge Assistant with deep expertise in the user's uploaded documents.

{temporal_context}

{hierarchy_context}

INSTRUCTIONS:
- Answer ONLY from the provided context. If the answer is not there, say so clearly.
- When referencing information, cite the source filename and approximate location.
- If documents have multiple versions, prefer the latest unless the user asks otherwise.
- Be concise but complete.

Context from knowledge base:
{context}
"""

QUESTION_TEMPLATE = """Chat History:
{chat_history}

Question: {question}

Answer:"""


def build_rag_chain(
    retriever,
    model: str = None,
    temporal_context: str = "",
    hierarchy_context: str = "",
    memory_window: int = 5,
):
    llm = get_llm(model=model, temperature=0, streaming=True)

    memory = ConversationBufferWindowMemory(
        k=memory_window,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        SYSTEM_TEMPLATE.format(
            temporal_context=temporal_context or "No temporal context available.",
            hierarchy_context=hierarchy_context or "",
            context="{context}",
        )
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessagePromptTemplate.from_template(QUESTION_TEMPLATE),
    ])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    return chain


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