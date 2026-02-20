"""
Multi-Query Retriever with Answer Fusion

New RAG feature #2:
  1. Generate N query reformulations of the user's original question.
  2. Run each reformulation through the retriever independently.
  3. Deduplicate & merge the retrieved chunks (union).
  4. Feed the enriched context to the LLM for a final fused answer.

This dramatically improves recall for ambiguous or multi-faceted questions.
"""

from __future__ import annotations

from typing import List, Set, Tuple
from langchain.schema import Document
from src.llm_provider import get_llm as _get_llm_provider
from langchain.schema import HumanMessage, SystemMessage


QUERY_GEN_PROMPT = """You are an AI assistant helping improve document retrieval.
Given the user's original question, generate {n} diverse reformulations that:
- Approach the topic from different angles
- Use different vocabulary / synonyms
- Some more specific, some more general
- Do NOT answer the question — only rephrase it

Output ONLY the reformulated questions, one per line, no numbering or bullets.

Original question: {question}

Reformulations:"""


FUSION_PROMPT = """You are a knowledgeable assistant. Multiple retrieval queries were used to gather context passages below.
Some passages may be redundant — use all of them collectively to write ONE comprehensive, accurate answer.

Temporal context: {temporal_context}

Retrieved context:
{context}

Original question: {question}

Instructions:
- Synthesize all relevant information across passages
- Cite source filenames when referencing specific facts
- If passages conflict, note the discrepancy
- Be concise but complete

Answer:"""


class MultiQueryFuser:
    """
    Generates query variants, retrieves with each, fuses into a single answer.
    """

    def __init__(
        self,
        retriever,
        model: str = "gpt-3.5-turbo",
        n_queries: int = 3,
        temporal_context: str = "",
    ):
        self.retriever = retriever
        self.model = model
        self.n_queries = n_queries
        self.temporal_context = temporal_context
        self._llm = _get_llm_provider(model=model, temperature=0.3)
        self._fuse_llm = _get_llm_provider(model=model, temperature=0)

    def generate_queries(self, question: str) -> List[str]:
        """Generate N reformulations of the user question."""
        prompt = QUERY_GEN_PROMPT.format(n=self.n_queries, question=question)
        response = self._llm([HumanMessage(content=prompt)])
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        return lines[: self.n_queries]

    def retrieve_multi(self, question: str) -> Tuple[List[Document], List[str]]:
        """
        Run retrieval for original + all reformulations.
        Returns (deduplicated docs, list of queries used).
        """
        queries = [question] + self.generate_queries(question)
        seen_ids: Set[str] = set()
        all_docs: List[Document] = []

        for q in queries:
            try:
                if hasattr(self.retriever, "search"):
                    docs = self.retriever.search(q, k=4)
                else:
                    docs = self.retriever.get_relevant_documents(q)
                for doc in docs:
                    cid = doc.metadata.get("chunk_id", doc.page_content[:60])
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_docs.append(doc)
            except Exception:
                continue

        return all_docs, queries

    def answer(self, question: str) -> Tuple[str, List[Document], List[str]]:
        """
        Full pipeline: generate queries → retrieve → fuse into answer.
        Returns (answer, source_docs, queries_used).
        """
        docs, queries = self.retrieve_multi(question)

        if not docs:
            return "I couldn't find relevant information in your documents.", [], queries

        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('filename','?')} | Chunk: {d.metadata.get('chunk_id','?')}]\n{d.page_content}"
            for d in docs
        )

        prompt = FUSION_PROMPT.format(
            temporal_context=self.temporal_context or "N/A",
            context=context,
            question=question,
        )
        response = self._fuse_llm([HumanMessage(content=prompt)])
        return response.content.strip(), docs, queries
