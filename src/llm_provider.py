"""
LLM Provider Factory.
Gemini: forces REST transport to avoid gRPC issues on Streamlit Cloud.
"""

from __future__ import annotations
import os
from typing import Literal
from src.logger import get_logger

log = get_logger("llm_provider")

Provider = Literal["openai", "gemini"]
_provider: Provider = "openai"
_api_key: str = ""


def configure_provider(provider: Provider, api_key: str):
    global _provider, _api_key
    _provider = provider
    _api_key = api_key
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
    log.info(f"Provider configured: {provider} | model family ready")


def get_provider() -> Provider:
    return _provider


def get_llm(model: str = None, temperature: float = 0, streaming: bool = False):
    if _provider == "gemini":
        return _GeminiRestLLM(
            model=model or "gemini-2.5-flash",
            temperature=temperature,
            api_key=_api_key or os.environ.get("GOOGLE_API_KEY", ""),
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name=model or "gpt-3.5-turbo",
            temperature=temperature,
            streaming=streaming,
        )


def get_embeddings():
    return _load_embeddings()


def _load_embeddings():
    """
    Load HuggingFace embeddings.
    Cached at app level via @st.cache_resource in app.py
    so the model is only downloaded once per session.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    log.info("Loading HuggingFace embeddings model (all-MiniLM-L6-v2)…")
    emb = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("Embeddings model loaded ✓")
    return emb


OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]


def available_models(provider: Provider) -> list:
    return GEMINI_MODELS if provider == "gemini" else OPENAI_MODELS


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _GeminiRestLLM:
    """Calls Gemini via REST API directly — no gRPC, no SDK transport issues."""

    def __init__(self, model: str, temperature: float, api_key: str):
        self.model_name = model
        self.temperature = temperature
        self.api_key = api_key

    def _call(self, prompt: str) -> str:
        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature}
        }
        resp = requests.post(url, json=payload, params={"key": self.api_key}, timeout=60)
        if not resp.ok:
            raise Exception(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def invoke(self, messages, **kwargs) -> _Msg:
        prompt = "\n\n".join(
            msg.content if hasattr(msg, "content") else str(msg)
            for msg in messages
        )
        return _Msg(self._call(prompt))

    def __call__(self, messages, **kwargs) -> _Msg:
        return self.invoke(messages, **kwargs)