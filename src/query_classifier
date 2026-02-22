"""
Query Classifier — detects intent before retrieval so the right prompt is used.

Intent types:
  summary     → "summarize", "overview", "tldr"
  comparison  → "compare", "difference", "vs"
  definition  → "what is", "define", "explain"
  factual     → specific fact questions
  quiz        → "quiz me", "test me", "questions"
  procedural  → "how to", "steps", "process"
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal
from .logger import get_logger

log = get_logger("query_classifier")

Intent = Literal["summary", "comparison", "definition", "factual", "quiz", "procedural"]

PATTERNS: list[tuple[Intent, list[str]]] = [
    ("summary",    ["summarize", "summarise", "summary", "overview", "tldr", "brief", "outline", "gist"]),
    ("comparison", ["compare", "comparison", "difference", "vs ", "versus", "contrast", "similarities", "better"]),
    ("quiz",       ["quiz me", "test me", "question me", "flashcard", "practice", "exam"]),
    ("procedural", ["how to", "how do", "steps to", "process of", "procedure", "walk me through", "guide me"]),
    ("definition", ["what is", "what are", "define", "definition", "explain what", "meaning of", "what does"]),
]

# Prompt templates per intent
PROMPTS: dict[Intent, str] = {
    "summary": """You are J.A.R.V.I.S. The user wants a summary. Provide a clear, structured summary using:
- Key points as bullet points
- Most important takeaways highlighted
- Organized sections if the content warrants it

Context:
{context}

Question: {question}

Structured Summary:""",

    "comparison": """You are J.A.R.V.I.S. The user wants a comparison. Structure your response as:
1. A markdown table comparing the key aspects
2. A brief narrative explaining the key differences
3. A recommendation or conclusion if applicable

Context:
{context}

Question: {question}

Comparison:""",

    "definition": """You are J.A.R.V.I.S. The user wants a definition or explanation. Provide:
- A clear, concise definition first (1-2 sentences)
- A more detailed explanation
- A concrete example from the documents if available

Context:
{context}

Question: {question}

Definition:""",

    "procedural": """You are J.A.R.V.I.S. The user wants to know how to do something. Provide:
- Numbered steps in order
- Any prerequisites or warnings
- Expected outcome

Context:
{context}

Question: {question}

Steps:""",

    "quiz": """You are J.A.R.V.I.S. The user wants to be tested. Generate 3 questions based on the context with answers.

Context:
{context}

Question: {question}

Quiz:""",

    "factual": """You are J.A.R.V.I.S. Answer the question directly and concisely using only the context below.
Cite the source filename when referencing specific facts.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:""",
}


@dataclass
class ClassifiedQuery:
    intent: Intent
    confidence: str   # "high" | "medium"
    prompt_template: str
    icon: str


_ICONS: dict[Intent, str] = {
    "summary":    "📑",
    "comparison": "⚔",
    "definition": "📖",
    "procedural": "🔧",
    "quiz":       "🎯",
    "factual":    "◈",
}


def classify(question: str) -> ClassifiedQuery:
    q = question.lower().strip()

    for intent, keywords in PATTERNS:
        for kw in keywords:
            if kw in q:
                log.info(f"Query classified as '{intent}' (keyword: '{kw}')")
                return ClassifiedQuery(
                    intent=intent,
                    confidence="high",
                    prompt_template=PROMPTS[intent],
                    icon=_ICONS[intent],
                )

    # Default to factual
    log.info("Query classified as 'factual' (default)")
    return ClassifiedQuery(
        intent="factual",
        confidence="medium",
        prompt_template=PROMPTS["factual"],
        icon=_ICONS["factual"],
    )