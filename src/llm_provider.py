"""
LLM Provider Factory — supports OpenAI and Google Gemini.
Embeddings use local sentence-transformers (free, no API quota).
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
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Use gemini-2.0-flash — works with both v1 and v1beta, free tier
        model = model or "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            convert_system_message_to_human=True,
        )
    else:
        from langchain_openai import ChatOpenAI
        model = model or "gpt-3.5-turbo"
        return ChatOpenAI(
            model_name=model,
            temperature=temperature,
            streaming=streaming,
        )


def get_embeddings():
    """Local embeddings — free, no API quota, works with any provider."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# gemini-2.0-flash is the current recommended free model
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]


def available_models(provider: Provider) -> list:
    return GEMINI_MODELS if provider == "gemini" else OPENAI_MODELS