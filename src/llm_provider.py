"""
LLM Provider Factory.
Gemini: uses google-generativeai SDK directly.
OpenAI: uses langchain-openai.
Embeddings: always local HuggingFace (free, no quota).
"""

from __future__ import annotations
import os
from typing import Literal

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


def get_provider() -> Provider:
    return _provider


def get_llm(model: str = None, temperature: float = 0, streaming: bool = False):
    if _provider == "gemini":
        return _GeminiDirectLLM(
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
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# gemini-2.5-flash is the model available on free tier keys
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]


def available_models(provider: Provider) -> list:
    return GEMINI_MODELS if provider == "gemini" else OPENAI_MODELS


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _GeminiDirectLLM:
    """Direct Gemini API call — bypasses langchain-google-genai entirely."""

    def __init__(self, model: str, temperature: float, api_key: str):
        self.model_name = model
        self.temperature = temperature
        self.api_key = api_key

    def _call(self, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        m = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={"temperature": self.temperature},
        )
        resp = m.generate_content(prompt)
        return resp.text

    def invoke(self, messages, **kwargs) -> _Msg:
        prompt = "\n\n".join(
            msg.content if hasattr(msg, "content") else str(msg)
            for msg in messages
        )
        return _Msg(self._call(prompt))

    def __call__(self, messages, **kwargs) -> _Msg:
        return self.invoke(messages, **kwargs)