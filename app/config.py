"""Runtime settings, read from environment variables (see .env.example)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "brand-brain"

    # Embeddings: Infinity serves an OpenAI-compatible /embeddings endpoint
    infinity_url: str = "http://infinity:7997"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024  # bge-m3 dense vector size

    # Local LLM: any OpenAI-compatible server (Ollama, llama.cpp server,
    # LM Studio, mlx_lm.server). Ollama exposes it under /v1.
    llm_base_url: str = "http://ollama:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    llm_api_key: str = "not-needed"  # local servers ignore it
    llm_timeout_s: float = 600.0  # CPU inference is slow; be patient

    # Retrieval
    retrieval_k: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Compress retrieved facts with the local LLM before answering.
    # Off by default: on a CPU-only box it added tens of seconds per request
    # (see README, "What didn't work").
    compress_context: bool = False

    answer_language: str = "English"


@lru_cache
def get_settings() -> Settings:
    return Settings()
