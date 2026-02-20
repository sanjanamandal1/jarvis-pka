"""
Context-aware RAG chain with temporal awareness.

The system prompt is dynamically built to include:
- Temporal context (document versions, upload dates, recent changes)
- Hierarchical document summaries for better answer grounding
- Source attribution with chunk IDs and filenames
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from src.llm_provider import get_llm
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document


SYSTEM_TEMPLATE = """You are a Personal Knowledge Assistant with deep expertise in the user's uploaded documents.

{temporal_context}

{hierarchy_context}

━━━ INSTRUCTIONS ━━━
• Answer ONLY from the provided context. If the answer isn't there, say so clearly.
• When referencing information, cite the source filename and approximate location.
• If documents have multiple versions, prefer the latest unless the user asks about a specific version.
• Highlight when information might be outdated based on document upload dates.
• Be concise but complete. Use bullet points for lists, prose for explanations.

Context from knowledge base:
{context}
"""

QUESTION_TEMPLATE = """Chat History:
{chat_history}

Question: {question}

Answer:"""


def build_rag_chain(
    retriever,
    model: str = "gpt-3.5-turbo",
    temporal_context: str = "",
    hierarchy_context: str = "",
    memory_window: int = 5,
):
    """
    Build a conversational RAG chain with temporal and hierarchical context.

    Parameters
    ----------
    retriever       : LangChain retriever from KnowledgeBase
    model           : OpenAI model name
    temporal_context: String from TemporalVersionManager.get_temporal_context()
    hierarchy_context: String from DocumentSummary.get_context_for_query()
    memory_window   : How many past exchanges to keep in context
    """
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
    """Format source documents for display in the UI."""
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
            "text_preview": doc.page_content[:280] + ("…" if len(doc.page_content) > 280 else ""),
            "token_count": doc.metadata.get("token_count", 0),
        })
    return sources
