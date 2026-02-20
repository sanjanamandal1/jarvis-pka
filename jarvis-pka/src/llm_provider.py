"""
LLM Provider Factory — supports OpenAI and Google Gemini.

Usage:
    from src.llm_provider import get_llm, get_embeddings, configure_provider

    configure_provider("gemini", api_key="AIza...")
    llm = get_llm("gemini-1.5-flash")
    embeddings = get_embeddings()
"""

from __future__ import annotations
import os
from typing import Literal

Provider = Literal["openai", "gemini"]

# Global provider state
_provider: Provider = "openai"


def configure_provider(provider: Provider, api_key: str):
    """Set the active provider and configure its API key."""
    global _provider
    _provider = provider

    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key


def get_provider() -> Provider:
    return _provider


def get_llm(model: str = None, temperature: float = 0, streaming: bool = False):
    """Return the appropriate LangChain LLM for the active provider."""
    if _provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            convert_system_message_to_human=True,  # Gemini quirk
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
    """Return the appropriate embeddings model for the active provider."""
    if _provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")


# ── Model name lists ──────────────────────────────────────────────────────────

OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]

GEMINI_MODELS = [
    "gemini-1.5-flash",      # fastest, cheapest — best for most use cases
    "gemini-1.5-pro",        # most capable Gemini
    "gemini-2.0-flash-exp",  # experimental latest
]

def available_models(provider: Provider) -> list:
    return GEMINI_MODELS if provider == "gemini" else OPENAI_MODELS
