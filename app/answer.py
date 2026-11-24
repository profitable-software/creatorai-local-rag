"""Answer drafting: retrieved context + question -> local LLM."""
import logging

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.llm import get_llm

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a content strategist helping a creator. Use the knowledge
below about their brand. Prefer it over general advice; if it does not cover the
question, say what is missing instead of guessing.
Answer in {language}."""

COMPRESS_PROMPT = """Select the facts relevant to the question and compress them into
one brief of at most 300 words. Keep names, numbers and audiences. Output only the
brief, in {language}."""


def format_context(docs: list[Document]) -> str:
    return "\n".join(f"- [{d.metadata.get('category') or d.metadata.get('data_type')}] {d.page_content}" for d in docs)


def build_messages(question: str, context: str, language: str) -> list[BaseMessage]:
    system = ANSWER_PROMPT.format(language=language)
    if context:
        system += f"\n\nKnowledge about this brand:\n{context}"
    return [SystemMessage(content=system), HumanMessage(content=question)]


async def compress(question: str, context: str, language: str) -> str:
    """Optional map-reduce step: shrink 30-50 facts into a short brief.

    Falls back to the raw context on any error, so a slow or failing local
    model never breaks the answer.
    """
    try:
        resp = await get_llm(temperature=0.1).ainvoke([
            SystemMessage(content=COMPRESS_PROMPT.format(language=language)),
            HumanMessage(content=f"Question: {question}\n\nFacts:\n{context}"),
        ])
        return resp.content.strip() or context
    except Exception:
        logger.warning("compression failed, using raw context", exc_info=True)
        return context


async def answer(question: str, docs: list[Document], language: str | None = None) -> str:
    s = get_settings()
    language = language or s.answer_language
    context = format_context(docs)
    if context and s.compress_context:
        context = await compress(question, context, language)
    resp = await get_llm(temperature=0.5).ainvoke(build_messages(question, context, language))
    return resp.content
