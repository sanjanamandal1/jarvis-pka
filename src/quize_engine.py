"""
Quiz Engine — generates MCQ and short-answer questions from document chunks.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional
from langchain_core.messages import HumanMessage
from src.llm_provider import get_llm

QUIZ_PROMPT = """You are a quiz generator. Given the document context below, generate {n} multiple choice questions.

Each question must test understanding of the content — not trivial facts.

Document context:
{context}

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "question": "Question text here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A",
    "explanation": "Brief explanation of why this is correct"
  }}
]"""


@dataclass
class QuizQuestion:
    question: str
    options: List[str]
    answer: str
    explanation: str
    user_answer: Optional[str] = None

    @property
    def is_correct(self) -> bool:
        if not self.user_answer:
            return False
        return self.user_answer.strip().upper()[0] == self.answer.strip().upper()[0]


@dataclass
class Quiz:
    questions: List[QuizQuestion] = field(default_factory=list)
    doc_name: str = ""
    score: int = 0
    completed: bool = False

    def calculate_score(self) -> int:
        self.score = sum(1 for q in self.questions if q.is_correct)
        return self.score


class QuizGenerator:
    def __init__(self, kb, model: str = None):
        self.kb = kb
        self.llm = get_llm(model=model, temperature=0.7)

    def generate(self, topic: str = "", n_questions: int = 5, doc_ids: list = None) -> Quiz:
        query = topic if topic else "key concepts important facts definitions"
        results = self.kb.search(query, k=8, doc_ids=doc_ids)

        docs = []
        for item in results:
            if isinstance(item, tuple):
                docs.append(item[0])
            elif hasattr(item, "page_content"):
                docs.append(item)

        if not docs:
            return Quiz(questions=[], doc_name=topic)

        context = "\n\n".join(
            f"[{d.metadata.get('filename','?')}]\n{d.page_content[:600]}"
            for d in docs[:6]
        )
        doc_name = docs[0].metadata.get("filename", "Documents") if docs else "Documents"

        prompt = QUIZ_PROMPT.format(n=n_questions, context=context)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(raw)
        except Exception:
            # Try to extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return Quiz(questions=[], doc_name=doc_name)
            else:
                return Quiz(questions=[], doc_name=doc_name)

        questions = []
        for item in data[:n_questions]:
            try:
                questions.append(QuizQuestion(
                    question=item["question"],
                    options=item["options"],
                    answer=item["answer"],
                    explanation=item.get("explanation", ""),
                ))
            except Exception:
                continue

        return Quiz(questions=questions, doc_name=doc_name)