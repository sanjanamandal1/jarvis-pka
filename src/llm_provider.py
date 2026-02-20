"""
LLM Provider Factory — calls Gemini directly via google-generativeai SDK.
This bypasses langchain-google-genai version issues completely.
"""

from __future__ import annotations
import os
from typing import Literal

Provider = Literal["openai", "gemini"]
_provider: Provider = "openai"


def configure_provider(provider: Provider, api_key: str):
    global _provider
    _provider = provider
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key


def get_provider() -> Provider:
    return _provider


def get_llm(model: str = None, temperature: float = 0, streaming: bool = False):
    if _provider == "gemini":
        return _GeminiDirectLLM(
            model=model or "gemini-1.5-flash",
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name=model or "gpt-3.5-turbo",
            temperature=temperature,
            streaming=streaming,
        )


def get_embeddings():
    """Local embeddings — free, no API quota."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]


def available_models(provider: Provider) -> list:
    return GEMINI_MODELS if provider == "gemini" else OPENAI_MODELS


# ── Direct Gemini wrapper (no langchain-google-genai needed) ──────────────────

class _GeminiMessage:
    """Minimal message object mimicking langchain AIMessage."""
    def __init__(self, content: str):
        self.content = content


class _GeminiDirectLLM:
    """
    Calls Gemini directly via google-generativeai SDK.
    Implements .invoke() to match LangChain interface.
    """

    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0):
        self.model = model
        self.temperature = temperature

    def invoke(self, messages, **kwargs) -> _GeminiMessage:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=api_key)

        # Convert messages to plain text prompt
        prompt_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                prompt_parts.append(msg.content)
            elif isinstance(msg, str):
                prompt_parts.append(msg)
        prompt = "\n\n".join(prompt_parts)

        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
            ),
        )
        response = model.generate_content(prompt)
        return _GeminiMessage(content=response.text)

    def __call__(self, messages, **kwargs) -> _GeminiMessage:
        return self.invoke(messages, **kwargs)