"""Local LLM client.

Everything talks to an OpenAI-compatible chat endpoint, so the same code runs
against Ollama (default), llama.cpp `llama-server`, LM Studio or `mlx_lm.server`
on Apple Silicon - only LLM_BASE_URL / LLM_MODEL change.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=temperature,
        timeout=s.llm_timeout_s,
    )
