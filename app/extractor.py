"""Knowledge extraction with a local LLM.

A transcript, article or chat log goes in; a list of small, self-contained,
categorised facts comes out. Facts (not raw chunks) are what makes the
"Brand Brain" useful across chats: they are short, de-duplicated by meaning,
and filterable by category.

7B-class models drift from JSON on long inputs, so the output is parsed
leniently and validated with Pydantic; bad items are dropped, not fatal.
"""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.llm import get_llm

logger = logging.getLogger(__name__)

CATEGORIES = (
    "target_audience", "tone_of_voice", "expertise", "strategy", "content_pillars",
    "key_values", "pain_points", "products", "competitors", "goals", "other",
)

SYSTEM_PROMPT = """You extract a knowledge base about a content creator's brand.
Read the text and return 5-30 standalone facts. Each fact must be understandable
without the source text, concrete (names, numbers, audiences, offers) and true to
the text - never invent anything.
Allowed categories: {categories}.
Write the facts in {language}.
Return ONLY JSON: {{"facts": [{{"fact": "...", "category": "...", "confidence": 0.0-1.0}}]}}"""


class Fact(BaseModel):
    fact: str = Field(min_length=3)
    category: str = "other"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_facts(raw: str) -> list[dict]:
    """Pull the first JSON object out of the model output and validate items."""
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        logger.warning("extractor: no JSON object in model output")
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("extractor: model output is not valid JSON")
        return []
    items = payload.get("facts", []) if isinstance(payload, dict) else []
    facts: list[dict] = []
    for item in items:
        try:
            fact = Fact.model_validate(item)
        except ValidationError:
            continue
        if fact.category not in CATEGORIES:
            fact.category = "other"
        facts.append(fact.model_dump())
    return facts


async def extract_facts(text: str, language: str | None = None) -> list[dict]:
    language = language or get_settings().answer_language
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(categories=", ".join(CATEGORIES), language=language)),
        HumanMessage(content=f"Text:\n\n{text}"),
    ]
    response = await get_llm(temperature=0.1).ainvoke(messages)
    return parse_facts(response.content)
