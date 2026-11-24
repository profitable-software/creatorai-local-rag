"""Unit tests that need no Qdrant, Infinity or LLM."""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import answer as answer_mod
from app import extractor
from app.answer import build_messages, format_context
from app.extractor import parse_facts
from app.ingest import fact_documents, split_text
from app.retriever import CHUNK, FACT, merge_unique, tenant_filter


def keys(flt):
    return [c.key for c in flt.must]


def test_tenant_filter_always_scopes_by_tenant():
    flt = tenant_filter("t1")
    assert keys(flt) == ["metadata.tenant_id"]
    assert flt.must[0].match.value == "t1"


def test_tenant_filter_with_type_and_categories():
    flt = tenant_filter("t1", data_type=FACT, categories=["goals", "products"])
    assert keys(flt) == ["metadata.tenant_id", "metadata.data_type", "metadata.category"]
    assert flt.must[2].match.any == ["goals", "products"]


def test_tenant_filter_rejects_empty_tenant():
    with pytest.raises(ValueError):
        tenant_filter("")


def test_merge_unique_keeps_order_and_drops_duplicates():
    a = [Document(page_content="x"), Document(page_content="y")]
    b = [Document(page_content="y"), Document(page_content="z")]
    assert [d.page_content for d in merge_unique(a, b)] == ["x", "y", "z"]


def test_split_text_tags_every_chunk():
    text = "word " * 800  # ~4000 chars -> several chunks
    chunks = split_text(text, "t1", "doc-1", "notes.md")
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata == {
            "tenant_id": "t1", "document_id": "doc-1", "source": "notes.md", "data_type": CHUNK,
        }
        assert len(c.page_content) <= 1000


def test_fact_documents_skips_empty_and_defaults_category():
    docs = fact_documents(
        [{"fact": "Audience: new runners", "category": "target_audience"}, {"fact": "  "}, {"fact": "x1", "category": None}],
        "t1", "doc-1", "notes.md",
    )
    assert [d.page_content for d in docs] == ["Audience: new runners", "x1"]
    assert docs[0].metadata["data_type"] == FACT
    assert docs[1].metadata["category"] == "other"


def test_parse_facts_handles_chatter_around_json():
    raw = 'Sure! Here you go:\n{"facts": [{"fact": "Sells a 6-week course", "category": "products", "confidence": 0.9}]}\nHope it helps.'
    assert parse_facts(raw) == [{"fact": "Sells a 6-week course", "category": "products", "confidence": 0.9}]


def test_parse_facts_drops_invalid_items_and_unknown_categories():
    raw = '{"facts": [{"fact": "ok fact", "category": "made_up"}, {"category": "goals"}, {"fact": "bad", "confidence": 7}]}'
    assert parse_facts(raw) == [{"fact": "ok fact", "category": "other", "confidence": 0.5}]


@pytest.mark.parametrize("raw", ["", "no json here", "{not json}", '["a list"]'])
def test_parse_facts_never_raises(raw):
    assert parse_facts(raw) == []


def test_build_messages_includes_context_and_language():
    ctx = format_context([Document(page_content="Tone: calm", metadata={"category": "tone_of_voice"})])
    msgs = build_messages("Write a hook", ctx, "Ukrainian")
    assert "[tone_of_voice] Tone: calm" in msgs[0].content
    assert "Ukrainian" in msgs[0].content
    assert msgs[1].content == "Write a hook"


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)


async def test_extract_facts_uses_local_llm(monkeypatch):
    fake = FakeLLM('{"facts": [{"fact": "Posts 3 reels a week", "category": "strategy"}]}')
    monkeypatch.setattr(extractor, "get_llm", lambda **_: fake)
    facts = await extractor.extract_facts("some transcript", language="English")
    assert facts[0]["fact"] == "Posts 3 reels a week"
    assert "some transcript" in fake.calls[0][1].content


async def test_compress_falls_back_to_raw_context(monkeypatch):
    monkeypatch.setattr(answer_mod, "get_llm", lambda **_: FakeLLM(error=TimeoutError()))
    assert await answer_mod.compress("q", "- raw facts", "English") == "- raw facts"
