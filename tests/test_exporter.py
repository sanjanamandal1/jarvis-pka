"""Tests for Exporter."""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exporter import Exporter


def make_chat_history():
    return [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI.",
         "intent": "factual", "intent_icon": "◈",
         "sources": [{"filename": "ml_intro.pdf", "version": 1}]},
        {"role": "user", "content": "Give me an example."},
        {"role": "assistant", "content": "Linear regression is a classic example.",
         "intent": "factual", "intent_icon": "◈", "sources": []},
    ]


def test_chat_to_markdown_contains_questions():
    history = make_chat_history()
    md = Exporter.chat_to_markdown(history, "Test Workspace")
    assert "What is machine learning?" in md
    assert "Machine learning is a subset of AI." in md
    assert "Test Workspace" in md


def test_chat_to_txt_is_plain():
    history = make_chat_history()
    txt = Exporter.chat_to_txt(history)
    assert "YOU:" in txt
    assert "JARVIS:" in txt
    assert "<" not in txt  # no HTML


def test_chat_to_markdown_has_sources():
    history = make_chat_history()
    md = Exporter.chat_to_markdown(history)
    assert "ml_intro.pdf" in md


def test_quiz_to_markdown():
    class FakeQuestion:
        question = "What is 2+2?"
        options = ["A. 3", "B. 4", "C. 5", "D. 6"]
        answer = "B"
        explanation = "Basic arithmetic."

    class FakeQuiz:
        questions = [FakeQuestion()]

    md = Exporter.quiz_to_markdown(FakeQuiz())
    assert "What is 2+2?" in md
    assert "Answer Key" in md
    assert "Basic arithmetic." in md
